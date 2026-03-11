# Changelog

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
