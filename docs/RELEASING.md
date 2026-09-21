# Releasing Synapse

## One-time setup (before the first tag)

### 1. Create the tap and bucket repos

Create two **public** repositories under `000Erick` on GitHub:

- `000Erick/homebrew-tap` — Homebrew formula tap (can be empty)
- `000Erick/scoop-bucket` — Scoop manifest bucket (can be empty)

### 2. Create a GitHub Personal Access Token

Create a **classic PAT** with `repo` scope (or a fine-grained token with **Contents: write** on both repos above). A single token can cover both repos.

### 3. Add the token as repository secrets on `synapse`

In `000Erick/engram-synapse` → Settings → Secrets and variables → Actions, add:

| Secret name | Value |
|---|---|
| `HOMEBREW_TAP_GITHUB_TOKEN` | The PAT from step 2 |
| `SCOOP_GITHUB_TOKEN` | The PAT from step 2 (can be the same token) |

`GITHUB_TOKEN` is provided automatically by GitHub Actions — no setup needed.

## Release flow

```sh
git tag v1.2.3
git push origin v1.2.3
```

That's it. The `release` workflow:

1. Builds binaries for **macOS, Linux, Windows** (amd64 + arm64) with `CGO_ENABLED=0`.
2. Injects the tag as the binary version via `-X main.version=v1.2.3`.
3. Attaches the six binaries + `checksums.txt` to the GitHub Release.
4. Pushes an updated Homebrew formula to `000Erick/homebrew-tap`.
5. Pushes an updated Scoop manifest to `000Erick/scoop-bucket`.

## End-user install options

```sh
# Homebrew (macOS / Linux)
brew install 000Erick/tap/synapse

# Scoop (Windows)
scoop bucket add 000Erick https://github.com/000Erick/scoop-bucket
scoop install synapse

# Go toolchain
go install github.com/000Erick/engram-synapse/cmd/synapse@latest

# Direct download
# Download the binary for your platform from:
# https://github.com/000Erick/engram-synapse/releases
```

## ⚠️ Important: tap/bucket repos must exist before the first tag

If `homebrew-tap` or `scoop-bucket` don't exist yet, or the token secrets are missing, `goreleaser` **will fail** the `brews` / `scoops` steps. Two options:

- **Complete steps 1–3 above** before pushing the first tag (recommended).
- **Binaries-only first release**: temporarily comment out the `brews:` and `scoops:` blocks in `.goreleaser.yaml`, tag, then restore them once the repos and secrets are in place.
