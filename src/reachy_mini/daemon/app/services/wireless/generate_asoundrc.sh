#!/bin/bash

# Output file path
OUTPUT_FILE="$HOME/.asoundrc"

# Get the line containing "Reachy Mini Audio" from aplay -l
CARD_LINE=$(aplay -l | grep -i "Reachy Mini Audio")

# Extract the card number (card X)
CARD_ID=$(echo "$CARD_LINE" | awk -F'card |:' '{print $2}' | tr -d ' ')

if [ -z "$CARD_ID" ]; then
    echo "Error: 'Reachy Mini Audio' sound card not found."
    exit 1
fi

# Write the content to the .asoundrc file
cat > "$OUTPUT_FILE" <<EOF
pcm.!default {
    type hw
    card $CARD_ID
}
ctl.!default {
    type hw
    card $CARD_ID
}
pcm.reachymini_audio_sink {
    type dmix
    ipc_key 4241
    slave {
        pcm "hw:$CARD_ID,0"
        channels 2
        period_size 1024
        buffer_size 4096
        rate 16000
    }
    bindings {
        0 0
        1 1
    }
}
pcm.reachymini_audio_src {
    type dsnoop
    ipc_key 4242
    slave {
        pcm "hw:$CARD_ID,0"
        channels 2
        rate 16000
        period_size 1024
        buffer_size 4096
    }
}
EOF

# Display success message
echo "File $OUTPUT_FILE successfully generated for Reachy Mini Audio sound card (card $CARD_ID)."
