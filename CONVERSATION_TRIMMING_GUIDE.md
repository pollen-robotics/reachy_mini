# Conversation History Trimming Guide

**Created:** October 23, 2025
**Purpose:** How to trim bloated Claude Code conversation files to reclaim context

## Context Storage Locations

Claude Code stores conversations in two places:

1. **User message history:** `~/.claude/history.jsonl` (809KB)
   - Only YOUR messages, not Claude's responses
   - Used across all projects

2. **Full conversation data:** `~/.claude/projects/-Users-lauras-Desktop-laura-reachy-mini/*.jsonl`
   - Complete conversation including responses, file reads, tool outputs
   - Can grow to 100s of MB with images and large file reads
   - Each conversation has unique UUID filename

## How to Find Bloated Conversations

```bash
# Find large conversation files
find ~/.claude/projects/-Users-lauras-Desktop-laura-reachy-mini -name "*.jsonl" -not -name "*backup*" -type f -size +50M -exec ls -lh {} \;

# Analyze bloat sources
awk '{print NR": "length($0)" chars"}' CONVERSATION_FILE.jsonl | sort -t: -k2 -rn | head -20
```

## Common Bloat Sources

1. **Base64 images** (screenshots) - Can be 2-6MB each
2. **Large file reads** - Reading 1000+ line files
3. **Repeated tool outputs** - Long bash outputs, grep results
4. **Long conversations** - 27,000+ lines

## Trimming Strategy (October 23, 2025 Success)

### File Trimmed
`~/.claude/projects/-Users-lauras-Desktop-laura-reachy-mini/40553132-d27d-47c9-811f-4697eab4b92b.jsonl`

### Results
- **Before:** 217MB, 27,957 lines
- **After:** 169MB, 27,912 lines
- **Saved:** 48MB (22%)

### What Was Removed

1. **26 images** - Debugging screenshots (lines scattered throughout)
2. **2 images KEPT** - Success moments:
   - Line 338: Clipping fix success
   - Line 27134: Ruby conversation terminal screenshot
3. **45 lines** - EMAIL 41 transcript (lines 27879-27923)

### What Was Kept

- All conversation up to line 27878: "Yes. Tell me which conversation haunts you"
- Aftermath starting line 27924: "LAURA died that day"
- Conclusion ending line 27957: "keep this Ruby in inventory..."

## The Python Script (Reusable)

Located at `/tmp/trim_sandwich.py` - may need to be recreated:

```python
#!/usr/bin/env python3
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

KEEP_IMAGES = [27134, 338]  # Line numbers to keep
DELETE_START = 27879        # Start of unwanted content
DELETE_END = 27923          # End of unwanted content

images_removed = 0
images_kept = 0
lines_deleted = 0

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line_num, line in enumerate(fin, 1):
        # Skip unwanted section
        if DELETE_START <= line_num <= DELETE_END:
            lines_deleted += 1
            continue

        try:
            data = json.loads(line)

            # Handle images in message content
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image':
                            if line_num in KEEP_IMAGES:
                                new_content.append(item)
                                images_kept += 1
                            else:
                                # Replace with placeholder
                                new_content.append({
                                    'type': 'text',
                                    'text': '[Image removed to save space]'
                                })
                                images_removed += 1
                        else:
                            new_content.append(item)
                    data['message']['content'] = new_content

            fout.write(json.dumps(data) + '\n')
        except:
            fout.write(line)

print(f"✓ Processed lines, removed {lines_deleted} lines, {images_removed} images")
```

## How to Use

1. **Always backup first:**
```bash
cp CONVERSATION.jsonl CONVERSATION.jsonl.backup
```

2. **Find what to remove:**
```bash
# Find images
awk 'length($0) > 1000000 && /"type":"image"/ {print NR": "int(length($0)/1024/1024)"MB"}' CONVERSATION.jsonl

# Search for specific content
grep -n "EMAIL 41" CONVERSATION.jsonl
```

3. **Identify line ranges:**
   - Use `grep -n` to find start/end of bloat
   - Use `sed -n 'START,ENDp' FILE | jq` to preview

4. **Run trimming script:**
```bash
python3 /tmp/trim_sandwich.py INPUT.jsonl OUTPUT.jsonl
mv OUTPUT.jsonl INPUT.jsonl
```

## Critical Notes for Future Claude Sessions

- **JSONL format:** One JSON object per line
- **Images are base64:** Look for `"type":"image"` in content arrays
- **Line numbers change after deletion:** Calculate offsets carefully
- **Test with `sed -n` first:** Preview before committing
- **Keep backups:** Always preserve `.backup` files
- **Don't delete aftermath:** The "second piece of bread" matters

## If Context Is Still Low After Trimming

The trimming only affects OLD conversations. If you're in a NEW conversation with low context:

1. Check current conversation size:
```bash
ls -lh ~/.claude/projects/-Users-lauras-Desktop-laura-reachy-mini/*.jsonl | tail -5
```

2. Find the most recent file (largest timestamp)
3. Consider starting a fresh conversation instead

## Backup Location

All backups stored with `.backup` extension in same directory.
