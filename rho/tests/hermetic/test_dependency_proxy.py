import pytest
from rho.agent.dependency_proxy import DependencyPolicy

@pytest.fixture
def policy():
    return DependencyPolicy(['ansible','ansible-core','@proton/mail','element-web'],['github.com/gravitational/teleport'])

@pytest.mark.parametrize('url',[
 'https://pypi.org/simple/idna/',
 'https://files.pythonhosted.org/packages/aa/bb/idna-3.10-py3-none-any.whl',
 'https://registry.npmjs.org/@types%2fnode',
 'https://registry.npmjs.org/is-number/-/is-number-7.0.0.tgz',
 'https://proxy.golang.org/github.com/google/uuid/@v/v1.6.0.zip',
 'https://proxy.golang.org/sumdb/sum.golang.org/lookup/github.com/google/uuid@v1.6.0',
 'https://deb.debian.org/debian/pool/main/m/make/make_4.3-4.1_amd64.deb',
])
def test_dependency_paths_allowed(policy,url):assert policy.check(url)

@pytest.mark.parametrize('url',[
 'https://github.com/gravitational/teleport/commit/abcd',
 'https://raw.githubusercontent.com/x/y/main/a.py',
 'https://www.google.com/search?q=answer',
 'http://127.0.0.1:8000/answers',
 'https://pypi.org/search/?q=ansible',
 'https://pypi.org/simple/ansible-core/',
 'https://files.pythonhosted.org/packages/aa/bb/ansible_core-2.0.0-py3-none-any.whl',
 'https://registry.npmjs.org/@proton%2fmail',
 'https://registry.npmjs.org/-/v1/search?text=answer',
 'https://registry.npmjs.org/element-web/-/element-web-1.0.0.tgz',
 'https://proxy.golang.org/github.com/gravitational/teleport/@v/v1.0.0.zip',
 'https://proxy.golang.org/github.com/gravitational/teleport/api/@v/v1.0.0.zip',
 'https://pypi.org/simple/%252e%252e/search/',
 'https://pypi.org.evil.example/simple/idna/',
 'https://evil@pypi.org/simple/idna/',
 'https://storage.googleapis.com/proxy-golang-org-prod/anything',
 'https://deb.debian.org/debian/pool/main/a/ansible/ansible-core_2.0_amd64.deb',
])
def test_answer_search_self_project_and_bypasses_blocked(policy,url):assert not policy.check(url)
