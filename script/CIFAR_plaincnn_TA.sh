layer=$1
dataset=cifar10
tmodel_name=resnet${1}_cifar
smodel_name=resnet${1}_cifar
seed=1
procedure=TA
aim=Correct_cnn_${dataset}_${smodel_name}_${layer}_$1_${procedure}_SEED_${seed}_DC_${2}_keepdownsampling
save_dir=./output/${dataset}/TA/plain_CNN_${smodel_name}/${aim}/
# data_dir=/cache/dataset/
data_dir=/home/zhfeing/datasets/cifar
model_dir=/home/zhfeing/new_projectX/distill/JointRD_Neurips2020/output/cifar10/resnet50_cifar/Correct_cnn_cifar10_resnet50_cifar_dirac_50_50_RES_NMT_SEED_1_DC_0.0_keepdownsampling/checkpoint/Correct_cnn_cifar10_resnet50_cifar_dirac_50_50_RES_NMT_SEED_1_DC_0.0_keepdownsampling_epoch_199_RES_NMT_checkpoint.pth.tar

dc=$2

###############

echo ${procedure}
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dirac.py --stage CNN_NMT \
                       --baseline_epochs 200 \
                       --cutout_length 0 \
                       --procedure ${procedure} \
                       --save_dir ${save_dir} \
                       --smodel_name ${smodel_name} \
                       --tmodel_name ${tmodel_name} \
                       --dataset ${dataset} \
                       --data_dir ${data_dir} \
                       --seed $seed \
                       --learning_rate 0.1 \
                       --batch_size 128 \
                       --aim ${aim} \
                       --start_epoch 0 \
                       --alpha 0.9 \
                       --weight_decay 5e-4 \
                       --kd_type margin \
                       --dis_weight 0. \
                       --lr_sch step \
                       --dc ${dc} \
                       --model=$model_dir
                    #    --model ${model_dir} \

                       
                       
                       
