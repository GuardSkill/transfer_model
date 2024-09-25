python sd2diffuser.py --scheduler_type ddim \
--original_config_file v1-inference.yaml \
--image_size 512 \
--checkpoint_path /Disk1/models/Stable-diffusion/Merge/PicLumen_H1.5_Anime_v0_15.safetensors \
--from_safetensors \
--to_safetensors \
--prediction_type epsilon \
--dump_path "ckpt_cache/PicLumen_H1.5_Anime_v0_15.safetensors"