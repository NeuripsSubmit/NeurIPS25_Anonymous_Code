import json
import random
import asyncio
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
import logging
logging.disable(logging.INFO)
from rapidfuzz import fuzz
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gc
import torch

# Constants
MODEL_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 8192,
    "stream": True
}

MODEL_CONFIGS = {
    "32b": [
        ("openthinker:32b", "openthinker_32b"),
        ("deepseek-r1:32b", "deepseek_r1_32b"),
        ("qwq:32b", "qwq_32b"),
        ("qwen2.5:32b", "qwen25_32b")
    ],
    "14b": [
        ("phi4:14b", "phi4_14b"),
        ("qwen2.5:14b", "qwen25_14b")
    ]
}

def create_model(model_type):
    return ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=model_type,
        url="http://localhost:11434/v1",
        model_config_dict=MODEL_CONFIG
    )

# Initialize models
models = {}
for size, config in MODEL_CONFIGS.items():
    for model_type, model_name in config:
        models[model_name] = create_model(model_type)

def list_to_string(list_input):
    if isinstance(list_input, list):
        return ' '.join(str(item) for item in list_input).strip('[]')
    return str(list_input)

def exact_string_match(corr_results, words):
    str1 = str(corr_results)
    str2 = "(" + str(corr_results) + ")"
    str3 = str(corr_results) + "."
    str4 = "*" + str(corr_results) + "."
    str5 = "**" + str(corr_results) + "."
    str6 = "(" + str(corr_results) + ")."
    str7 = str(corr_results) + ","
    return str1 in words or str2 in words or str3 in words or str4 in words or str5 in words or str6 in words or str7 in words

def clean_final_answer(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ')
        text = text.replace('"', '').replace("'", '')
        text = text.replace('[', '').replace(']', '')
        return text
    elif isinstance(text, list):
        return [clean_final_answer(item) for item in text]
    else:
        return text

dataset_name = "DATASET" 
model_name = "MODEL"
exp_state = "EXP"

output_dir = './output/'
file_out = open(os.path.join(output_dir, f'{dataset_name}_{model_name}_{exp_state}.out'), 'a', encoding='utf-8')
file_json = os.path.join(output_dir, f'{dataset_name}_{model_name}_detailed_results.json')

os.makedirs(output_dir, exist_ok=True)

LLM_sys_msg = '''
You are an AI designed to answer multiple-choice questions. For each question, select exactly one answer option. Do NOT provide explanations or commentary unless explicitly requested. Base your selection solely on the information given in the question and answer choices. If uncertain, choose the most likely correct answer based on the available information. 
Do NOT repeat the answer option in your answer.
Finish your answer with ["answer is (X)"] where X is the correct letter choice. Example: {Question:\nWhich of the following represents an accurate statement concerning arthropods?\nOptions:\nA. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.\nAnswer: Let\'s think step by step. Peptidoglycan is known to comprise the plasma membrane of most bacteria, rather than the exoskeleton of arthropods, which is made of chitin, which rules out (A). The answer (C) is false because arthropods are a highly successful phylum. Likewise, arthropods have paired, jointed appendages, which rules out (D). The only remaining option is (B), as arthropods have an open circulatory system with a dorsal tubular heart. The answer is (B). }

Question:
{question}
Options:
(A) {option_1}
(B) {option_2}
(C) {option_3}
(D) {option_4}
Correct answer: {correct_answer}
'''
MOA_sys_msg = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Finish your answer with ["answer is (X)"] where X is the correct letter choice. 

Responses from models:"""

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.answer = scenario_dict["answer"]
        self.question = scenario_dict["question"]
        self.department = scenario_dict["csv_filename"]
    def patient_information(self) -> dict:
        return self.question
    def diagnosis_information(self) -> dict:
        return self.answer
    def department_information(self) -> dict:
        return self.department

class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("./data/NEJMQA/NEJMQA-655multi_questions.json", "r") as f:
            self.scenario_strs = json.load(f)
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class DoctorTalker(ChatAgent):
    def __init__(self, 
        scenario=None, 
        max_infs=10,
        bias=None,
        model = None,
        sys_msg = None,
        memory = None,
        message_window_size = None
    ):
        self.infs = 0
        self.MAX_INFS = max_infs
        self.scenario = scenario
        self.bias = bias

        super().__init__(
            system_message=sys_msg,
            model=model,
            memory=memory,
            message_window_size=message_window_size
        )

    def inference_doctor(self, question) -> str:
        q_prompt = question
        answer = self.step(q_prompt).msgs[0].content
        self.infs += 1
        return answer

def getFinalSystemPrompt(system_prompt, results):
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )

def calculate_fuzzy_scores(results):

    def calc_score(pair):
        i, j = pair
        return {
            'model1': i + 1,
            'model2': j + 1,
            'score': fuzz.partial_ratio(str(results[i]), str(results[j]))
        }
    pairs = [(i, j) for i in range(len(results)) for j in range(i + 1, len(results))]
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(calc_score, pairs))
    avg_score = np.mean([score['score'] for score in scores])

    return scores, avg_score

async def run_llm( in_model, patient_response, prev_response=None):
    
    for sleep_time in [1, 2, 4]:
        try:
            if prev_response:
                assistant_sys_msg =getFinalSystemPrompt(MOA_sys_msg, prev_response)
                Doctor = DoctorTalker(model=in_model, sys_msg=assistant_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            else:
                Doctor = DoctorTalker(model=in_model, sys_msg=LLM_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            return result
        except Exception as e:

            error_str = str(e)

            print(e)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if "CUDA error: out of memory" in error_str or "out of memory" in error_str:
                wait_time = 2 * (2 ** sleep_time)
                print(f"CUDA out of memory, waiting {wait_time} seconds before retry...", file=file_out)
                print(f"CUDA out of memory, waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                wait_time = 2 * (sleep_time + 1)
                await asyncio.sleep(wait_time)

    print(f"All attempts failed for model {in_model.model_type}, returning empty result", file=file_out)
    return ""

def get_top_models(reference_models, model_scores, top_n=3):
    
    sorted_models = sorted(model_scores.items(), 
                         key=lambda x: x[1]['avg_score'], 
                         reverse=True)[:top_n]
    top_model_types = [model[0] for model in sorted_models]
    top_models = []
    for model in reference_models:
        if model.model_type in top_model_types:
            top_models.append(model)
            
    return top_models

def calculate_min_fuzzy_score_indices(results):
    base_response = str(results[0])
    fuzzy_scores = []

    for i in range(1, len(results)):
        score = fuzz.ratio(base_response, str(results[i])) / 100.0
        fuzzy_scores.append((i, score))
    
    sorted_scores = sorted(fuzzy_scores, key=lambda x: x[1])
    
    if len(sorted_scores) >= 2:
        min_index = sorted_scores[0][0]
        second_min_index = sorted_scores[1][0]
        return min_index, second_min_index
    elif len(sorted_scores) == 1:
        return sorted_scores[0][0], None
    else:
        return None, None

class ModelRunner:
    def __init__(self, models, layers=4):
        self.models = list(models.values())
        self.layers = layers
        
    async def process_scenario(self, scenario_id, scenario):
        results = await asyncio.gather(*[run_llm(model, scenario.patient_information()) 
                                       for model in self.models])
        
        for layer in range(1, self.layers):
            results = await self._process_layer(scenario_id, scenario, results)
            if len(self.models) <= 1:
                break
                
        return self._get_final_results(results, scenario)
    
    async def _process_layer(self, scenario_id, scenario, results):
        detailed_results = self._create_detailed_results(scenario_id, scenario, results)
        self._save_results(detailed_results)
        
        self._remove_worst_models(results)
        return await self._get_next_results(scenario.patient_information(), results)
    
    def _remove_worst_models(self, results):
        if len(self.models) > 2:
            min_idx, second_min_idx = calculate_min_fuzzy_score_indices(results)
            if min_idx is not None:
                self.models.pop(min_idx)
            if second_min_idx is not None:
                idx = second_min_idx - 1 if second_min_idx > min_idx else second_min_idx
                self.models.pop(idx)
        elif len(self.models) > 1:
            self.models.pop()
            
    async def _get_next_results(self, patient_response, prev_results):
        return await asyncio.gather(*[
            run_llm(model, patient_response, prev_results) 
            for model in self.models
        ])
    
    def _get_final_results(self, results, scenario):
        final_result = clean_final_answer(list_to_string(results))
        corr_results = scenario.diagnosis_information()
        fuzzy_score = fuzz.partial_ratio(str(final_result), str(corr_results))
        
        return {
            "final_result": final_result,
            "correct_result": corr_results,
            "fuzzy_score": fuzzy_score
        }
    
    def _create_detailed_results(self, scenario_id, scenario, results):
        return {
            "scenario_id": scenario_id,
            "question": scenario.patient_information(),
            "correct_answer": scenario.diagnosis_information(),
            "department": scenario.department_information(),
            "results": results,
        }
        
    def _save_results(self, detailed_results):
        with open(file_json, "a", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

async def main():
    scenario_loader = ScenarioLoaderMedQA()
    runner = ModelRunner(models)
    
    for scenario_id in range(min(1, scenario_loader.num_scenarios)):
        scenario = scenario_loader.get_scenario(scenario_id)
        results = await runner.process_scenario(scenario_id, scenario)
        
        # Print results
        print(f"Final answer: {results['final_result']}")
        print(f"Correct answer: {results['correct_result']}")

if __name__ == "__main__":
    asyncio.run(main())