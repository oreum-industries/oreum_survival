#!/usr/bin/env bash
# hash_verbatim_files.sh

# loops over files in local directory `verbatim`
# creates an SHA1 hash of each
# write the hashes to a local file with today's date

# today=$(date +'%Y-%m-%d-%H%M%S')
# echo "${today}"

fn="hashes_of_verbatim_files.txt"
touch $fn
echo $(date) >> "${fn}"

hashAll() {
    for i in "$1"/*; do
        if [ -f "$i" ]; then                 # regular file
            echo "$i"
            openssl sha1 "$i" >> "$fn"
        elif [ -d "$i" ]; then                # dir
            hashAll "$i"
        fi
    done
    echo "" >> "${fn}"
}
hashAll "verbatim"
