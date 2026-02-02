#!/bin/bash
# PostToolUse hook: solvers/ や features/ のPythonファイルが編集されたら
# 簡単な構文チェックを実行する
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.file_path // empty')

if [[ "$FILE_PATH" == *.py ]] && [[ "$FILE_PATH" == *solvers/* || "$FILE_PATH" == *features/* || "$FILE_PATH" == *metrics/* || "$FILE_PATH" == *policy/* ]]; then
    python -c "import py_compile; py_compile.compile('$FILE_PATH', doraise=True)" 2>&1
    if [ $? -ne 0 ]; then
        echo "Syntax error in $FILE_PATH" >&2
        exit 2
    fi
fi

exit 0
