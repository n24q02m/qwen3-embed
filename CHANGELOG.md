# Changelog

## [0.2.1-beta](https://github.com/n24q02m/qwen3-embed/compare/v0.2.0...v0.2.1-beta) (2026-02-14)


### Features

* add Q4F16 ONNX and GGUF model variant support ([8cc81c0](https://github.com/n24q02m/qwen3-embed/commit/8cc81c0ed10a3968758ba252dbc948aedc16cab1))


### Documentation

* update README with Q4F16 and GGUF variants ([c17f975](https://github.com/n24q02m/qwen3-embed/commit/c17f97541d579bda276b7138ed650deb1e7561d3))

## [0.2.0](https://github.com/n24q02m/qwen3-embed/compare/v0.1.0...v0.2.0) (2026-02-13)


### ⚠ BREAKING CHANGES

* Package import changed from 'from fastembed import ...' to 'from qwen3_embed import ...'. The PyPI package name remains qwen3-embed.

### Features

* promote dev to main (v0.2.0-beta) ([#2](https://github.com/n24q02m/qwen3-embed/issues/2)) ([4c95738](https://github.com/n24q02m/qwen3-embed/commit/4c95738222df7ae0ea339dc07d2a56d973e0365b))
* qwen3-embed v0.1.0 - trimmed fastembed fork for Qwen3 models only ([d2a53fc](https://github.com/n24q02m/qwen3-embed/commit/d2a53fce71b55c44071717bd6722bcaf6fbcef7a))

## [0.2.0-beta](https://github.com/n24q02m/qwen3-embed/compare/v0.1.0...v0.2.0-beta) (2026-02-13)


### ⚠ BREAKING CHANGES

* Package import changed from 'from fastembed import ...' to 'from qwen3_embed import ...'. The PyPI package name remains qwen3-embed.

### Features

* qwen3-embed v0.1.0 - trimmed fastembed fork for Qwen3 models only ([d2a53fc](https://github.com/n24q02m/qwen3-embed/commit/d2a53fce71b55c44071717bd6722bcaf6fbcef7a))


### Bug Fixes

* port ONNX compatibility fixes and add integration tests ([59e30d0](https://github.com/n24q02m/qwen3-embed/commit/59e30d00138668921763592d357ea68efb0decc1))
* resolve CI lint failures and exclude integration tests from CI ([a645d52](https://github.com/n24q02m/qwen3-embed/commit/a645d524a3f0b322e62fad3baf758995c19decc9))


### Code Refactoring

* rename package from fastembed to qwen3_embed ([6e77ba5](https://github.com/n24q02m/qwen3-embed/commit/6e77ba5715d48e33fe8bb863a135e17b2a385a1a))

## 0.1.0 (2026-02-13)

### Features

* Qwen3TextEmbedding: last-token pooling + MRL support (32-1024 dims), instruction-aware queries
* Qwen3CrossEncoder: causal LM yes/no logit scoring with chat template
* ONNX models downloaded from HuggingFace Hub at runtime
* Forked from qdrant/fastembed, trimmed to Qwen3 models only
