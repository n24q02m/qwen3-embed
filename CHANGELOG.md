# Changelog

## [1.3.0](https://github.com/n24q02m/qwen3-embed/compare/v1.2.0...v1.3.0) (2026-03-11)

### Features

* add unit tests for remove_non_alphanumeric utility ([#223](https://github.com/n24q02m/qwen3-embed/issues/223)) ([c991044](https://github.com/n24q02m/qwen3-embed/commit/c991044))
* add tests for DenseModelDescription dimension limits ([#188](https://github.com/n24q02m/qwen3-embed/issues/188)) ([b204339](https://github.com/n24q02m/qwen3-embed/commit/b204339))
* add coverage for add_extra_session_options ([#187](https://github.com/n24q02m/qwen3-embed/issues/187)) ([d3e998c](https://github.com/n24q02m/qwen3-embed/commit/d3e998c))
* add test for _collect_file_metadata function ([#186](https://github.com/n24q02m/qwen3-embed/issues/186)) ([3731db9](https://github.com/n24q02m/qwen3-embed/commit/3731db9))
* add tests for get_all_punctuation utility ([#185](https://github.com/n24q02m/qwen3-embed/issues/185)) ([d8422d1](https://github.com/n24q02m/qwen3-embed/commit/d8422d1))

### Bug Fixes

* revert Python to 3.13, fix Renovate config, add Q4F16/GGUF integration tests, fix model cache validation ([3770fe8](https://github.com/n24q02m/qwen3-embed/commit/3770fe8))
* improve metadata save error logging and add test ([#224](https://github.com/n24q02m/qwen3-embed/issues/224)) ([d84cd31](https://github.com/n24q02m/qwen3-embed/commit/d84cd31))
* improve decompress_to_cache TarError handling test ([#193](https://github.com/n24q02m/qwen3-embed/issues/193)) ([00a48ba](https://github.com/n24q02m/qwen3-embed/commit/00a48ba))
* suppress Bandit B615 on offline snapshot_download ([#222](https://github.com/n24q02m/qwen3-embed/issues/222)) ([306f3b8](https://github.com/n24q02m/qwen3-embed/commit/306f3b8))
* add MD5 hash verification for GCS file downloads ([#221](https://github.com/n24q02m/qwen3-embed/issues/221)) ([3582fbc](https://github.com/n24q02m/qwen3-embed/commit/3582fbc))
* fix insecure temporary directory creation ([#212](https://github.com/n24q02m/qwen3-embed/issues/212)) ([a2ec057](https://github.com/n24q02m/qwen3-embed/commit/a2ec057))
* fix arbitrary file write via archive extraction (Tar Slip) ([#218](https://github.com/n24q02m/qwen3-embed/issues/218)) ([ce7a251](https://github.com/n24q02m/qwen3-embed/commit/ce7a251))
* fix unsafe file download via unvalidated URL (SSRF) ([#203](https://github.com/n24q02m/qwen3-embed/issues/203)) ([35df610](https://github.com/n24q02m/qwen3-embed/commit/35df610))
* standardize CI with PR title check, email notify, and templates ([a4582b5](https://github.com/n24q02m/qwen3-embed/commit/a4582b5))

### Security

* fix path traversal via archive extraction (Tar Slip) ([#218](https://github.com/n24q02m/qwen3-embed/issues/218))
* fix SSRF via unvalidated URL in file downloads ([#203](https://github.com/n24q02m/qwen3-embed/issues/203))
* fix insecure temporary directory creation ([#212](https://github.com/n24q02m/qwen3-embed/issues/212))
* add MD5 hash verification for GCS file downloads ([#221](https://github.com/n24q02m/qwen3-embed/issues/221))
* fix model cache validation (false cache hit with Q4F16 variant) ([3770fe8](https://github.com/n24q02m/qwen3-embed/commit/3770fe8))

### Documentation

* update docs for stable release - Production/Stable status, complete CHANGELOG, accurate README and AGENTS.md ([dae3f2f](https://github.com/n24q02m/qwen3-embed/commit/dae3f2f))

## [1.2.0](https://github.com/n24q02m/qwen3-embed/compare/v1.1.3...v1.2.0) (2026-03-04)

### Features

* migrate to 2025-2026 tech stack (uv/ty) ([7b689e4](https://github.com/n24q02m/qwen3-embed/commit/7b689e4))
* add Codecov coverage upload ([891201b](https://github.com/n24q02m/qwen3-embed/commit/891201b))
* add Renovate for automated dependency updates ([4e307b4](https://github.com/n24q02m/qwen3-embed/commit/4e307b4))
* add comprehensive test coverage for utilities and components ([f1d2997](https://github.com/n24q02m/qwen3-embed/commit/f1d2997))

### Bug Fixes

* make TextCrossEncoder.add_custom_model case-insensitive ([#88](https://github.com/n24q02m/qwen3-embed/issues/88)) ([30212f0](https://github.com/n24q02m/qwen3-embed/commit/30212f0))
* fix path traversal and DoS risk ([#118](https://github.com/n24q02m/qwen3-embed/issues/118)) ([1bf8a49](https://github.com/n24q02m/qwen3-embed/commit/1bf8a49))
* increase test coverage to 92% and resolve typing errors ([7658adf](https://github.com/n24q02m/qwen3-embed/commit/7658adf))

## [1.1.3](https://github.com/n24q02m/qwen3-embed/compare/v1.1.2...v1.1.3) (2026-02-28)

### Bug Fixes

* correct model identifiers from Qwen/ to n24q02m/ namespace ([298405c](https://github.com/n24q02m/qwen3-embed/commit/298405c))

## [1.1.2](https://github.com/n24q02m/qwen3-embed/compare/v1.1.1...v1.1.2) (2026-02-27)

### Bug Fixes

* GGUF Reranker Device.AUTO defaulting to CPU instead of GPU ([684af12](https://github.com/n24q02m/qwen3-embed/commit/684af12))

## [1.1.1](https://github.com/n24q02m/qwen3-embed/compare/v1.1.0...v1.1.1) (2026-02-25)

### Bug Fixes

* auto-detect GPU for GGUF backend (Device.AUTO uses n_gpu=-1) ([edb58bb](https://github.com/n24q02m/qwen3-embed/commit/edb58bb))

### Documentation

* add GPU acceleration section and fix CPU-only claim ([add0204](https://github.com/n24q02m/qwen3-embed/commit/add0204))

## [1.1.0](https://github.com/n24q02m/qwen3-embed/compare/v1.0.0...v1.1.0) (2026-02-22)

### Features

* auto-detect DirectML GPU provider and improve logging ([f400012](https://github.com/n24q02m/qwen3-embed/commit/f400012))

## [1.0.0](https://github.com/n24q02m/qwen3-embed/compare/v0.2.1...v1.0.0) (2026-02-18)

### Chores

* migrate from release-please to python-semantic-release v10 ([9a78411](https://github.com/n24q02m/qwen3-embed/commit/9a78411))

## [0.2.1](https://github.com/n24q02m/qwen3-embed/compare/v0.2.0...v0.2.1) (2026-02-14)

### Features

* add Q4F16 ONNX and GGUF model variant support ([8cc81c0](https://github.com/n24q02m/qwen3-embed/commit/8cc81c0))

### Documentation

* update README with Q4F16 and GGUF variants ([c17f975](https://github.com/n24q02m/qwen3-embed/commit/c17f975))

## [0.2.0](https://github.com/n24q02m/qwen3-embed/compare/v0.1.0...v0.2.0) (2026-02-13)

### ⚠ BREAKING CHANGES

* Package import changed from `from fastembed import ...` to `from qwen3_embed import ...`. The PyPI package name remains `qwen3-embed`.

### Features

* qwen3-embed v0.1.0 - trimmed fastembed fork for Qwen3 models only ([d2a53fc](https://github.com/n24q02m/qwen3-embed/commit/d2a53fc))

### Bug Fixes

* port ONNX compatibility fixes and add integration tests ([59e30d0](https://github.com/n24q02m/qwen3-embed/commit/59e30d0))

### Code Refactoring

* rename package from fastembed to qwen3_embed ([6e77ba5](https://github.com/n24q02m/qwen3-embed/commit/6e77ba5))

## 0.1.0 (2026-02-13)

### Features

* Qwen3TextEmbedding: last-token pooling + MRL support (32-1024 dims), instruction-aware queries
* Qwen3CrossEncoder: causal LM yes/no logit scoring with chat template
* ONNX models downloaded from HuggingFace Hub at runtime
* Forked from qdrant/fastembed, trimmed to Qwen3 models only
