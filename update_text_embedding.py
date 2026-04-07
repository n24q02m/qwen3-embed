file_path = "qwen3_embed/text/text_embedding.py"
with open(file_path) as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "class TextEmbedding(TextEmbeddingBase):" in line:
        new_lines.append(line)
        new_lines.append(
            "    _embedding_type_cache: dict[str, type[TextEmbeddingBase]] | None = None\n"
        )
        new_lines.append(
            "    _embedding_description_cache: dict[str, DenseModelDescription] | None = None\n"
        )
        continue

    if "registered_models = cls._list_supported_models()" in line:
        new_lines.append("        cls._ensure_cache_populated()\n")
        continue

    if "for registered_model in registered_models:" in line:
        new_lines.append("        if model_lower in cls._embedding_description_cache:\n")
        # Skip original loop
        continue

    if "if model_lower == registered_model.model.lower():" in line:
        continue

    if "CustomTextEmbedding.add_model(" in line:
        new_lines.append("        cls._clear_registry_cache()\n")
        new_lines.append(line)
        continue

    if (
        "model_name_lower = model_name.lower()" in line
        and "def __init__" in lines[lines.index(line) - 10 : lines.index(line)]
    ):
        new_lines.append("        cls._ensure_cache_populated()\n")
        new_lines.append(line)
        continue

    if "for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:" in line:
        new_lines.append("        if model_name_lower in self._embedding_type_cache:\n")
        new_lines.append(
            "            EMBEDDING_MODEL_TYPE = self._embedding_type_cache[model_name_lower]\n"
        )
        new_lines.append("            self.model = EMBEDDING_MODEL_TYPE(\n")
        new_lines.append("                model_name=model_name,\n")
        new_lines.append("                cache_dir=cache_dir,\n")
        new_lines.append("                threads=threads,\n")
        new_lines.append("                providers=providers,\n")
        new_lines.append("                cuda=cuda,\n")
        new_lines.append("                device_ids=device_ids,\n")
        new_lines.append("                lazy_load=lazy_load,\n")
        new_lines.append("                **kwargs,\n")
        new_lines.append("            )\n")
        new_lines.append("            return\n")
        # We need to skip the original loop.
        # This is getting complicated with line-by-line. Let's use a better approach.
        pass

with open(file_path, "w") as f:
    f.writelines(lines)  # Restore
