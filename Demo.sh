# AI config
      
python test.py   --flow_type 'flownet2.0'   --input_data_path 'AI32' \
                 --mode_type 'AI' \
                 --test_model_DB_path     './pretrained_models/pretrained_model_AI/pretrained_model_AI32_L2.pth'\
                 --save_txt_path          './Test_result/Test_result_AI32_L2.txt'\
                 --save_result_path       './Rec_frame/Rec_frame_AI32_L2'\
                 --gpu_id 2 



# LD config
: '  
python test.py     --flow_type 'flownet2.0' --input_data_path 'LD32' \
                   --mode_type 'LD' \
                   --test_model_DB_path     './pretrained_models/pretrained_model_LD/pretrained_model_LD32_L2.pth'\
                   --save_txt_path          './Test_result/Test_result_LD32_L2.txt'\
                   --save_result_path       './Rec_frame/Rec_frame_LD32_L2'\
                   --gpu_id 2
'
echo "Compiling ....."