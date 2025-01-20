# run the shell script from the main directory
# sudo bash src/tools/merl_workflow/create_folder.sh

create_folder(){
    mkdir -p output/merl/merl_$1
    mkdir -p output/generation

    cd output/merl/merl_$1/

    if [ $1 -le 6 ] 
    then 
        # create a folder for each file in the list
        while IFS= read -r line; do
            mkdir -p $line
        done < ../../../data/merl/merl_dataset_list.txt
        ls | wc -l
    else
        # Linear interpolation merl_7 --> 0  (-7)
        while IFS=' ' read -r BRDF1 BRDF2 weights; do
            mkdir -p $BRDF1"_"$BRDF2"_"$weights
        done < ../../../data/merl/interpolate/interpolation_$(((index-7)/6)).txt
        ls | wc -l
    fi 

    cd ../../../
}
 

for index in {1..24}
do
    echo $index
    # echo $(((index-7)/6))
    create_folder $index
    sudo chmod -R 777 ./output/merl/merl_$index
done