"""Audited HTTP(S) dependency proxy over a per-container Unix socket.

CONNECT is terminated with an ephemeral, container-trusted CA so HTTPS paths
are checked too. No opaque tunnels, arbitrary websites, search APIs or current
project packages. The host CA key is never mounted into a task container.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import re
import socket
import ssl
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit, urljoin
import httpx

POLICY_VERSION = 'dependencies-only-v1'
HOSTS = {
    'pypi.org','files.pythonhosted.org','registry.npmjs.org','registry.yarnpkg.com',
    'proxy.golang.org','sum.golang.org','storage.googleapis.com',
    'deb.debian.org','security.debian.org','archive.ubuntu.com','security.ubuntu.com',
    'dl-cdn.alpinelinux.org','nodejs.org','go.dev','dl.google.com',
}


def normalized(name): return re.sub(r'[-_.]+','-',name.lower())


class DependencyPolicy:
    def __init__(self, projects=(), modules=()):
        self.projects = {normalized(p) for p in projects if p}
        self.modules = {m.lower().rstrip('/') for m in modules if m}

    def package_allowed(self, name):
        name = normalized(name)
        return name not in self.projects and name.rsplit('/',1)[-1] not in self.projects

    def file_allowed(self, name):
        name = normalized(name)
        return not any(name.startswith(p+'-') and name[len(p)+1:len(p)+2].isdigit()
                       for p in self.projects)

    def check(self, url, *, redirect=False):
        u=urlsplit(url)
        if u.scheme not in {'http','https'} or u.username or u.password or u.port not in {None,80,443}:
            return False
        host=(u.hostname or '').lower()
        if host not in HOSTS: return False
        path=u.path
        for _ in range(4):
            decoded=unquote(path)
            if decoded==path: break
            path=decoded
        if '%' in path or '\\' in path or any(x in {'.','..'} for x in path.split('/')):return False
        if any(ord(c)<32 for c in path):return False
        if u.query and host!='storage.googleapis.com':return False
        if host=='pypi.org':
            m=re.fullmatch(r'/simple/([A-Za-z0-9_.-]+)/?',path)
            return bool(m and self.package_allowed(m[1]))
        if host=='files.pythonhosted.org':
            return path.startswith('/packages/') and bool(re.search(r'\.(whl|tar\.gz|tar\.bz2|zip|metadata)$',path)) and self.file_allowed(path.rsplit('/',1)[-1])
        if host in {'registry.npmjs.org','registry.yarnpkg.com'}:
            m=re.fullmatch(r'/(@[\w.-]+/[\w.-]+|[\w.-]+)(?:/(?:[\w.+~^-]+|-/[\w.+-]+\.tgz))?/?',path)
            return bool(m and self.package_allowed(m[1]))
        if host=='proxy.golang.org':
            if path.startswith('/sumdb/sum.golang.org/'):
                return bool(re.fullmatch(r'/sumdb/sum\.golang\.org/(supported|latest|lookup/[\w./!@+~-]+|tile/[\w./!-]+)',path))
            m=re.fullmatch(r'/(.+)/(?:@v/(?:list|[\w.+~!-]+\.(?:info|mod|zip))|@latest)',path)
            if not m:return False
            module=re.sub(r'!([a-z])',lambda x:x[1].upper(),m[1]).lower()
            return not any(module==p or module.startswith(p+'/') for p in self.modules)
        if host=='sum.golang.org':
            return bool(re.fullmatch(r'/(latest|lookup/[\w./!@+~-]+|tile/[\w./!-]+)',path))
        if host=='storage.googleapis.com':
            # Signed storage objects are accepted only as redirects from an
            # already-approved module request, never as direct client targets.
            return redirect and path.startswith('/proxy-golang-org-prod/')
        if host in {'deb.debian.org','security.debian.org','archive.ubuntu.com','security.ubuntu.com'}:
            return bool(re.fullmatch(r'/(debian(?:-security)?|ubuntu)/(dists/[\w/+.~-]+|pool/[\w/+.~-]+\.deb)',path)) and self.file_allowed(path.rsplit('/',1)[-1].split('_')[0]+'-0')
        if host=='dl-cdn.alpinelinux.org':
            return bool(re.fullmatch(r'/alpine/[\w./+-]+/(APKINDEX\.tar\.gz|[\w.+-]+\.apk)',path)) and self.file_allowed(path.rsplit('/',1)[-1])
        if host in {'go.dev','dl.google.com'}:
            return bool(re.fullmatch(r'/(dl/|go/)go[0-9.]+\.(linux|darwin|windows)-[\w]+\.(tar\.gz|zip)(\.sha256)?',path))
        if host=='nodejs.org':
            return bool(re.fullmatch(r'/dist/v[0-9.]+/(node-v[\w.+-]+\.(tar\.(xz|gz)|zip)|SHASUMS256\.txt(?:\.sig|\.asc)?)',path))
        return False


class DependencyProxy(AbstractContextManager):
    def __init__(self, socket_path:Path, state_dir:Path, policy=None, *, static_resources=None):
        self.socket_path=socket_path;self.state_dir=state_dir
        self.policy=policy or DependencyPolicy();self.lock=threading.Lock();self.cert_lock=threading.Lock()
        self.contexts={};self.ready=False
        # Explicit grader-only dependency fixtures. Solver callers never set
        # these; exact URLs are served from pinned bytes without network access.
        self.static_resources = dict(static_resources or {})
        self.static_hosts = {urlsplit(url).hostname for url in self.static_resources}

    def set_projects(self,projects,modules):
        self.policy=DependencyPolicy(projects,modules);self.ready=True

    def request(self, method, url, headers):
        for attempt in range(3):
            try:
                response=self.client.send(self.client.build_request(method,url,headers=headers),stream=True)
                if response.status_code not in {429,502,503,504} or attempt==2:return response
                response.close()
            except httpx.HTTPError:
                if attempt==2:raise
            time.sleep(attempt+1)

    def log(self,url,allowed,status):
        with self.lock:
            with (self.state_dir/'requests.jsonl').open('a') as f:
                f.write(json.dumps({'time':time.time(),'url':url,'allowed':allowed,'status':status})+'\n')

    def cert_context(self,host):
        with self.cert_lock:
            if host in self.contexts:return self.contexts[host]
            stem=self.state_dir/('leaf-'+hashlib.sha256(host.encode()).hexdigest()[:16])
            ext=stem.with_suffix('.ext');ext.write_text(f'subjectAltName=DNS:{host}\nextendedKeyUsage=serverAuth\n')
            def run(args):subprocess.run(['openssl',*args],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            run(['req','-new','-newkey','rsa:2048','-nodes','-keyout',str(stem.with_suffix('.key')),
                 '-out',str(stem.with_suffix('.csr')),'-subj','/CN='+host])
            run(['x509','-req','-in',str(stem.with_suffix('.csr')),'-CA',str(self.ca),'-CAkey',str(self.key),
                 '-set_serial',str(int(hashlib.sha256(host.encode()).hexdigest()[:16],16)),
                 '-days','3','-out',str(stem.with_suffix('.crt')),'-extfile',str(ext)])
            ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.set_alpn_protocols(['http/1.1'])
            ctx.load_cert_chain(stem.with_suffix('.crt'),stem.with_suffix('.key'))
            self.contexts[host]=ctx;return ctx

    def __enter__(self):
        self.state_dir.mkdir(parents=True,exist_ok=True);self.state_dir.chmod(0o700)
        self.ca=self.state_dir/'ca.crt';self.key=self.state_dir/'ca.key'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','3',
                        '-keyout',str(self.key),'-out',str(self.ca),'-subj','/CN=RHO dependency gateway',
                        '-addext','basicConstraints=critical,CA:TRUE'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.key.chmod(0o600)
        self.client=httpx.Client(timeout=httpx.Timeout(180,connect=30),follow_redirects=False)
        owner=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            tunnel_host=None
            def log_message(self,*_):pass
            def reject(self,url,reason='dependency_policy_denied'):
                owner.log(url,False,reason)
                body=(reason+'\n').encode();self.send_response(403)
                self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close');self.end_headers()
                self.wfile.write(body);self.close_connection=True
            def do_CONNECT(self):
                authority=urlsplit('https://'+self.path)
                host=authority.hostname or ''
                if not owner.ready or host not in HOSTS | owner.static_hosts or authority.port!=443 or authority.username or authority.path:
                    return self.reject('CONNECT '+self.path)
                self.send_response(200,'Connection established');self.end_headers();self.wfile.flush()
                try:
                    self.connection=owner.cert_context(host).wrap_socket(self.connection,server_side=True)
                    self.rfile=self.connection.makefile('rb',-1);self.wfile=self.connection.makefile('wb',0)
                    self.tunnel_host=host
                    self.close_connection=False
                    self.handle_one_request()
                except (ssl.SSLError,OSError):pass
                finally:self.close_connection=True
            def do_GET(self):self.forward()
            def do_HEAD(self):self.forward()
            def do_POST(self):self.reject(self.path,'only_dependency_GET_HEAD_allowed')
            def forward(self):
                url=('https://'+self.tunnel_host+self.path) if self.tunnel_host and self.path.startswith('/') else self.path
                parsed=urlsplit(url)
                if (not owner.ready or (url not in owner.static_resources and not owner.policy.check(url)) or
                    (self.tunnel_host and parsed.hostname!=self.tunnel_host)):
                    return self.reject(url)
                if url in owner.static_resources:
                    body = owner.static_resources[url]
                    owner.log(url, True, 'static_dependency_200')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Connection', 'close'); self.end_headers()
                    if self.command != 'HEAD': self.wfile.write(body)
                    self.close_connection = True
                    return
                headers={'Accept-Encoding':'identity','User-Agent':'rho-dependency-gateway/1'}
                if self.headers.get('Range'):headers['Range']=self.headers['Range']
                if parsed.hostname in {'registry.npmjs.org','registry.yarnpkg.com'}:
                    headers['Accept']='application/vnd.npm.install-v1+json'
                elif parsed.hostname=='pypi.org':headers['Accept']='text/html'
                target=url
                try:
                    for _ in range(6):
                        response=owner.request(self.command,target,headers)
                        if response.status_code not in {301,302,303,307,308}:break
                        next_url=urljoin(target,response.headers.get('location',''));response.close()
                        if not owner.policy.check(next_url,redirect=True):return self.reject(next_url,'redirect_denied')
                        target=next_url
                    else:return self.reject(url,'redirect_limit')
                    owner.log(url,True,response.status_code)
                    self.send_response(response.status_code)
                    for k,v in response.headers.items():
                        if k.lower() not in {'connection','transfer-encoding','set-cookie','strict-transport-security'}:
                            self.send_header(k,v)
                    self.send_header('Connection','close');self.end_headers()
                    try:
                        for chunk in response.iter_raw():self.wfile.write(chunk)
                    finally:response.close()
                except (httpx.HTTPError,OSError) as exc:
                    owner.log(url,True,type(exc).__name__)
                    try:self.send_error(502,'Dependency upstream failure')
                    except OSError:pass
                finally:self.close_connection=True
        class Server(ThreadingHTTPServer):
            address_family=socket.AF_UNIX
            daemon_threads=True
            def server_bind(self):
                self.socket.bind(self.server_address);self.server_name='localhost';self.server_port=0
        self.server=Server(str(self.socket_path),Handler);self.socket_path.chmod(0o666)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        return self

    def __exit__(self,*exc):
        self.server.shutdown();self.server.server_close();self.thread.join();self.client.close()
        self.socket_path.unlink(missing_ok=True)
