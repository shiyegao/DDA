export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 4 --num_samples 4 --timestep_respacing 100 \
                            --model_path ckpt/256x256_diffusion_uncond.pt --base_samples dataset/imagenetc \
                            --down_N 8 --range_t 20 --start_step 50 \
                            --corruption gaussian_noise --severity 5 \
                            --save_dir dataset/generated/