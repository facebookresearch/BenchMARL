#!/bin/bash

# Default values
OUTPUT_FILE="Project_contents.md"
EXCLUDE_DIRS="${EXCLUDE_DIRS:-".git,pdf_files,results,.vscode,temp"}"
# Default extensions if none provided via -i
# INCLUDE_EXTS="${INCLUDE_EXTS:-"txt,yaml,yml,py,sh,md,js,json"}"
INCLUDE_EXTS="${INCLUDE_EXTS:-"txt,yaml,yml,py,sh,js,json"}"

usage() {
    echo "Usage: $0 [options]"
    echo "  -i, --include: Comma-separated extensions (e.g., py,yaml)"
    echo "  -e, --exclude: Comma-separated directories to skip"
    echo "  -o, --output:  Target markdown file (default: output.md)"
    echo "  -h, --help:  Display this help"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--include) INCLUDE_EXTS="$2"; shift ;;
        -e|--exclude) EXCLUDE_DIRS="$2"; shift ;;
        -o|--output)  OUTPUT_FILE="$2"; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

echo "# Project Documentation Summary" > "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"

# Prepare extension array for find
IFS=',' read -ra EXTS <<< "$INCLUDE_EXTS"
FIND_EXT_ARGS=()
for ext in "${EXTS[@]}"; do
    [[ ${#FIND_EXT_ARGS[@]} -gt 0 ]] && FIND_EXT_ARGS+=("-o")
    FIND_EXT_ARGS+=("-name" "*.$ext")
done

# Prepare exclusion array
IFS=',' read -ra EXCLS <<< "$EXCLUDE_DIRS"
EXCLUDE_ARGS=()
for dir in "${EXCLS[@]}"; do
    EXCLUDE_ARGS+=("-not" "-path" "*/$dir/*")
done

# Process files
find . -type f "${EXCLUDE_ARGS[@]}" \( "${FIND_EXT_ARGS[@]}" \) | while read -r file; do
    # Skip if file is ignored by git (if in a git repo)
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        if git check-ignore -q "$file"; then
            continue
        fi
    fi

    # Formatting and output
    clean_path="${file#./}"
    ext="${file##*.}"
    
    # Map extension to markdown language
    case "$ext" in
        py) lang="python" ;;
        yaml|yml) lang="yaml" ;;
        sh) lang="bash" ;;
        js) lang="javascript" ;;
        json) lang="json" ;;
        md) lang="markdown" ;;
        *) lang="$ext" ;;
    esac

    {
        echo -e "\n## $clean_path"
        echo "\`\`\`$lang"
        cat "$file"
        echo -e "\n\`\`\`"
        echo ""
    } >> "$OUTPUT_FILE"
done

echo "Process complete. Output saved to $OUTPUT_FILE"
