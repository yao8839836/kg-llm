# Edge MAE

The implementation of KG-LLM in our paper: 

Exploring Large Language Models for Knowledge Graph Completion


## Installing requirement packages

```bash
pip install -r requirements_chatglm.txt
```

## Data

(1) The four KGs we used as well as entity and relation descriptions are in ./data.

(2) The input files for LLMs are also in each folder of ./data, see train_instructions_llama.json and train_instructions_glm.json as examples. 

(3) The output files of our models are also in each folder of ./data, see pred_instructions_llama13b.csv and generated_predictions.txt (from ChatGLM-6B) as examples.

## How to run
 
### 1. LLaMA fine-tuning and inference examples

```shell
python lora_finetune_wn11.py
```
```shell
python lora_finetune_wn11.py
```

```shell
python lora_infer_wn11.py
```
```shell
python lora_infer_yago_rel.py
```

### 2. ChatGLM fine-tuning and inference examples

```shell
python ptuning_main.py --do_train --train_file data/YAGO3-10/train_instructions_glm_rel.json --validation_file data/YAGO3-10/test_instructions_glm_rel.json --prompt_column prompt --response_column response --overwrite_cache --model_name_or_path models/chatglm-6b --output_dir models/yago-rel-chatglm-6b --overwrite_output_dir --max_source_length 230 --max_target_length 20 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --predict_with_generate --max_steps 80000 --logging_steps 300 --save_steps 10000 --learning_rate 1e-2 --pre_seq_len 8 --quantization_bit 4
```

```shell
python ptuning_main.py --do_predict --validation_file data/YAGO3-10/test_instructions_glm_rel.json --test_file data/YAGO3-10/test_instructions_glm_rel.json --overwrite_cache --prompt_column prompt --response_column response --model_name_or_path models/yago-rel-chatglm-6b/checkpoint-10000 --output_dir /data/YAGO3-10/glm_r_result --overwrite_output_dir --max_source_length 230 --max_target_length 20 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 8 --quantization_bit 4
```

Change the --model_name_or_path from models/yago-rel-chatglm-6b/checkpoint-10000 to the original model path models/chatglm-6b is for original ChatGLM-6B inference.

### 3. Raw LLaMA inference

```shell
python test_llama_fb13.py
```

### 4. Generate input files for LLMs
 
Please see instructions_XX.py, human_FB13_data.py and human_YAGO3_data.py.

### 5. Evaluation

Please see eval_XX_ft.py, human_FB13_eval.py, human_YAGO3_eval_XX.py