FILES="./summary/*/*"
for f in $FILES
do
	echo "Processing $f"
    python -m export_data_as_csv ./summary/test-3/0111202656 --write-csv --no-write-pkl --out-dir "$f.csv"

done