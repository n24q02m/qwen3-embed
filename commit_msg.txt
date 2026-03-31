🧪 Improve test coverage for get_all_punctuation with extended Unicode

🎯 What: Added test coverage for `get_all_punctuation` to explicitly verify that it includes non-ASCII, extended Unicode punctuation marks (such as em-dashes, quotation marks, and various other typographical symbols).

📊 Coverage:
- `test_contains_extended_punctuation` now covers:
  - `—` (Em dash)
  - `“` / `”` (Double quotation marks)
  - `¡` / `¿` (Inverted exclamation/question marks)
  - `«` / `»` (Angle quotation marks)
  - `…` (Horizontal ellipsis)

✨ Result: Ensured `get_all_punctuation` correctly categorizes broad Unicode punctuation logic via the `unicodedata.category` check.
