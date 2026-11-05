#!/bin/bash
# Publishes a locally built image to the static registry on dev-vm, so that
#   docker pull dockerstatic.tsuku.re/gpubox:latest
# works anywhere.
#
#   ./publish.sh                         push localhost/gpubox:latest as :latest
#   ./publish.sh localhost/gpubox:x a b  push that image as tags a and b
#
# The registry is plain files served by nginx (see the tsukure infra repo,
# nginx/dev-vm/dockerstatic.tsuku.re.conf). Layers are uploaded first and only
# when missing, then the manifest, then the tags, so a pull never sees a tag
# whose layers are still uploading.
set -euo pipefail

image=${1:-localhost/gpubox:latest}
shift || true
tags=("$@")
[ ${#tags[@]} -gt 0 ] || tags=(latest)

repo=${REPO:-gpubox}
remote=${REMOTE:-dev-vm}
root=${ROOT:-/srv/dockerstatic}
dest="$root/v2/$repo"

work=$(mktemp -d "${TMPDIR:-/var/tmp}/gpubox-publish.XXXXXX")
trap 'rm -rf "$work"' EXIT

echo "== exporting $image (gzip layers, Docker v2 schema 2)"
podman push --quiet --format v2s2 --compress --compression-format gzip "$image" "dir:$work/dir"

manifest="$work/dir/manifest.json"
mdigest="sha256:$(sha256sum "$manifest" | cut -d' ' -f1)"
mapfile -t blobs < <(jq -r '.config.digest, .layers[].digest' "$manifest")

mkdir -p "$work/tree/blobs" "$work/tree/manifests"
size=0
for d in "${blobs[@]}"; do
    f="$work/dir/${d#sha256:}"
    [ "sha256:$(sha256sum "$f" | cut -d' ' -f1)" = "$d" ] || { echo "digest mismatch for $d" >&2; exit 1; }
    ln "$f" "$work/tree/blobs/$d"
    size=$((size + $(stat -c %s "$f")))
done
cp "$manifest" "$work/tree/manifests/$mdigest"
echo "   manifest $mdigest, ${#blobs[@]} blobs, $((size / 1048576)) MiB compressed"

ssh "$remote" "mkdir -p '$dest/blobs' '$dest/manifests'"

echo "== uploading missing blobs"
rsync -a --info=progress2 --partial --ignore-existing "$work/tree/blobs/" "$remote:$dest/blobs/"

echo "== uploading manifest"
rsync -a --ignore-existing "$work/tree/manifests/" "$remote:$dest/manifests/"

echo "== tagging ${tags[*]}"
for t in "${tags[@]}"; do
    [[ $t =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || { echo "bad tag: $t" >&2; exit 1; }
    ssh "$remote" "ln -sfn '$mdigest' '$dest/manifests/.$t.new' && mv -Tf '$dest/manifests/.$t.new' '$dest/manifests/$t'"
done

host=${PUBLIC_HOST:-dockerstatic.tsuku.re}
echo "== done"
for t in "${tags[@]}"; do echo "   docker pull $host/$repo:$t"; done
