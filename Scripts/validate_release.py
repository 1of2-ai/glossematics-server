#!/usr/bin/env python3
"""Check the public Helmet release contract without executing downloaded code."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile


def validate(directory, tag):
    root = Path(directory)
    release = json.loads((root / 'release.json').read_text())
    version = release['version']
    assert re.fullmatch(r'\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?', version), 'Invalid version'
    assert tag == 'v' + version, 'Tag and manifest version differ'
    assert release['architecture'] == 'arm64', 'Expected arm64'
    assert release['team_identifier'] == 'KA589LJT76', 'Expected 1of2 signing team'
    assert release['notarized'] is True, 'Only notarized builds may be published'
    assert re.fullmatch(r'[a-f0-9]{40}', release['source_commit']), 'Missing source commit'
    assert isinstance(release['server'], dict) and release['server'], 'Missing server description'
    models = release['models']
    assert models, 'Missing model catalog'
    for model in models:
        for key in ('id', 'label', 'repository', 'revision', 'size_bytes', 'dimensions', 'default_dimensions'):
            assert key in model, f'Model missing {key}'
    base = f'gloss-server-{version}-macos-arm64'
    assets = ['release.json']
    for suffix, key in (('.tar.gz', 'archive'), ('.dmg', 'disk_image')):
        name = base + suffix
        assert release[key] == name, f'Unexpected {key}'
        hasher = hashlib.sha256()
        with (root / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        fields = (root / (name + '.sha256')).read_text().split()
        assert fields == [digest, name], f'Invalid checksum for {name}'
        if key == 'archive':
            assert digest == release['archive_sha256'], 'Archive checksum differs from manifest'
        assets.extend((name, name + '.sha256'))
    with tarfile.open(root / release['archive'], 'r:gz') as archive:
        members = archive.getmembers()
        names = {m.name.rstrip('/'): m for m in members}
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts, 'Unsafe archive path'
            assert path.parts[0] == base, 'Unexpected archive root'
            assert member.isfile() or member.isdir() or member.issym(), 'Unsupported archive member'
            if member.issym():
                assert member.name == base + '/gloss-server' and member.linkname == 'glossematicsd', 'Unexpected symlink'
        daemon = names[base + '/glossematicsd']
        assert daemon.isfile() and daemon.mode & 0o111, 'Daemon is not executable'
        compat = names[base + '/gloss-server']
        assert compat.issym() and compat.linkname == 'glossematicsd', 'Missing Helmet compatibility entry'
        assert any(n.startswith(base + '/Glossematics_glossematicsd.bundle/') for n in names), 'Missing resources'
    return assets


if __name__ == '__main__':
    try:
        print('\n'.join(validate(sys.argv[1], sys.argv[2])))
    except (AssertionError, KeyError, ValueError, OSError, IndexError, tarfile.TarError) as error:
        sys.exit(f'Release validation failed: {error}')
