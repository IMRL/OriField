fold=$1; scene=$2; 
for f in `gsutil -m ls "gs://waymo_open_dataset_v_2_0_0/$fold/"`; do arr=(${f//\// }); echo ${arr[3]}; mkdir -p $fold/${arr[3]}/; gsutil -m cp $f$scene $fold/${arr[3]}/$scene; done
