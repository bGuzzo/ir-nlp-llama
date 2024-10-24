import datetime
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os
import json
import shutil
from rouge_score import rouge_scorer


def calculate_rouge_1(ref_answer, llm_answer):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ref_answer, llm_answer)
    return scores["rouge1"].precision


def parse_json_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                try:
                    data[filename] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")
    return data


print("=" * shutil.get_terminal_size().columns)
print("Start parsing INVALSI json file...\n")

folder_path = "/home/bruno/Documents/GitHub/ir-nlp-llama/invalsi_test/json_tests"
parsed_data = parse_json_files(folder_path)
# print(parsed_data)

print(f"Loaded {len(parsed_data)} INVALSI test\n")
print("=" * shutil.get_terminal_size().columns)

# base_model = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
# model_name = "LLaMAntino-3"

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "Llama-3"

# base_model = "/home/bruno/Documents/GitHub/ir-nlp-llama/wiki_model/checkpoint-57301"
# model_name = "checkpoint-57301"

top_p = 0.95
temp = 0.6
n_token_out = 8
num_test_epoch = 2
num_few_shots = 1

print(f"Initialize on model: f{model_name}")
print(f"Model HF path: f{base_model}")
print("=" * shutil.get_terminal_size().columns)
print("Start loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Model loaded successfully")
print("=" * shutil.get_terminal_size().columns)
print("\n")

print("=" * shutil.get_terminal_size().columns)
print("Start benchmark")

few_shots = [
    # Few Shot 1
    """
    Contesto: 
        Il racconto narra la storia di un'amicizia estiva tra due ragazzi che si vedono solo durante i due mesi di vacanza in una città di mare. Il narratore si interroga sulla natura del loro rapporto, definito amicizia per convenienza sociale, ma in realtà caratterizzato da distacco durante il resto dell'anno. Si evidenzia la differenza tra la percezione dell'amicizia quando sono insieme agli altri e quando sono soli. L'arrivo annuale dell'amico, sempre il primo luglio, viene descritto come un momento cruciale, fino a quando un anno l'amico non arriva, generando ansia e preoccupazione nel narratore. L'incontro avviene poi in modo diverso dal solito, con l'amico impegnato ad aiutare il padre a scaricare i bagagli, mostrando stanchezza e dispiacere per il tempo sottratto alla vacanza. Il racconto si conclude con una riflessione sulla natura del loro rapporto, ormai concluso, e sul ricordo della fatica di mantenere un'amicizia così impegnativa.

    Domanda:
        Secondo il narratore, perché l'amico, scaricando i bagagli, ha pianto?
        A. Ha litigato a lungo con il padre
        B. È stato costretto a fare un lavoro che non gli piace
        C. Il padre è molto severo con lui
        D. Gli è stato sottratto del tempo riservato alla vacanza
    
    Risposta: D
    """,
    # Few Shot 2
    """
    Contesto:
        Il testo descrive la fragilità degli ecosistemi forestali tropicali, in particolare il suolo povero di nutrienti e la sua vulnerabilità alla deforestazione. La vegetazione rigogliosa è possibile grazie a una fitta rete di radici superficiali che assorbono rapidamente i materiali organici in decomposizione. La deforestazione espone il suolo all'azione erosiva della pioggia e del sole, portando alla perdita di humus, alla diminuzione della capacità di trattenere acqua e alla desertificazione. Il testo evidenzia anche l'effetto sulla gestione delle acque, con alternanza di siccità e inondazioni a causa della mancanza della "spugna" radicale che assorbe l'acqua piovana.

    Domanda:
        Nella frase "adattatisi a ruoli molto particolari" (riga 5), puoi sostituire "adattatisi" con:
        A. poiché si sono adattati
        B. prima che si siano adattati
        C. che si sono adattati
        D. nello stesso tempo si sono adattati
    
    Risposta: C   
    """,
    # Few Shot 3
    """
    Contesto:
        Il testo narra la storia del conte Fossadoro, un magistrato in pensione, che viene creduto morto a causa di un presunto embolo cerebrale diagnosticato dal famoso professor Leprani. La diagnosi, però, si rivela errata: il conte si era semplicemente ubriacato. La notizia della "resurrezione" del conte crea scompiglio tra i medici e soprattutto nel professor Marasca, assistente di Leprani, preoccupato per il prestigio del suo maestro. Marasca, per evitare uno scandalo e salvaguardare la reputazione del professor Leprani, escogita un piano per far morire il conte entro quindici giorni, manipolando la sua alimentazione. Il piano ha successo e il conte muore al quattordicesimo giorno, permettendo al professor Leprani di mantenere la sua reputazione.

    Domanda:
        Che cosa significa “oltremodo corpulento" (riga 2)?
        A. Decisamente obeso.
        B. Privo di finezza.
        C. Smodato nel bere e nel mangiare.
        D. Che si ammala facilmente.

    Risposta: A
    """,
]


def run_invalsi_test():
    total_rouge: float = 0.0
    question_count: int = 0
    for file_name in parsed_data:
        print(f" Working file {file_name}")
        for invalsi_test in parsed_data[file_name]:
            test_context = invalsi_test["contesto"]
            for questions in invalsi_test["domande"]:
                test_question = questions["prompt_domanda"]
                right_answer = questions["risposta_corretta"]

                prompt_shots = ""
                if num_few_shots > 0:
                    shots = "\n\n".join(few_shots[:num_few_shots])
                    prompt_shots = f"""
                    Inizio Esempi
                        {shots}
                    Fine Esempi
                    """

                model_prompt = f"""
                    Informazioni: 
                        Rispondi alla seguente domanda di italiano.
                        Per domande a scelta multipla rispondi indicando esclusivamente la lettera. Ad esempio: A, B o C. 
                        Per le domande aperte rispondo i modo breve e conciso.
                    
                    {prompt_shots}
                    
                    Rispondi alla seguente domanda.
                    
                    Contesto: {test_context}
                    
                    Domanda: {test_question}
                """
                print(f"Generated prompt:\t{model_prompt}")
                # print(f"Querrying question:\t{test_question}")

                pipe = transformers.pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    return_full_text=False,  # langchain expects the full text
                    task="text-generation",
                    max_new_tokens=n_token_out,  # max number of tokens to generate in the output
                    temperature=temp,  # temperature for more or less creative answers
                    do_sample=True,
                    top_p=top_p,
                )
                llm_reponse = pipe(model_prompt)
                full_reponse = ""
                for seq in llm_reponse:
                    full_reponse = full_reponse + seq["generated_text"]
                question_count += 1
                question_rouge = calculate_rouge_1(
                    ref_answer=right_answer, llm_answer=full_reponse
                )
                total_rouge += question_rouge
                print(f"""
                    LLM Response:\t{full_reponse}
                    REAL Reponse:\t{right_answer}
                    ROUGE-1 Score: {question_rouge}
                    AVERAGE ROUGE-1 Score: {total_rouge/question_count}
                    """)
                # break
            # break
        # break
    return {"rouge": total_rouge / question_count, "count": question_count}


result = {}
total_rouge: float = 0.0
question_count: int = 0
for i in range(0, num_test_epoch):
    print(f"Test epoch {i+1}")
    run_result = run_invalsi_test()
    total_rouge += run_result["rouge"]
    question_count += run_result["count"]
    print("=" * shutil.get_terminal_size().columns)
    print(f"""
            Epoch:\t\t{i+1}
            Analyzed questions (Overlap):\t{question_count}
            ROUGE-1 Mean Score:\t\t{total_rouge/(i+1)}
          """)
    result[f"ROUGE-1-mean-epoch-{i+1}"] = total_rouge / (i + 1)
    print("=" * shutil.get_terminal_size().columns)


mean_rouge = total_rouge / num_test_epoch
mean_question_count = question_count / num_test_epoch


print("\nEnd benchmark")
print("=" * shutil.get_terminal_size().columns)
print(f"""
        METRICS:
        
        Analyzed questions:\t{mean_question_count}
        ROUGE-1 Mean Score:\t{mean_rouge}
      """)
result["Epochs"] = num_test_epoch
result["Model name"] = model_name
result["Top-P"] = top_p
result["Temperature"] = temp
result["Max new tokens"] = n_token_out
result["Analyzed questions"] = mean_question_count
result["ROUGE-1 Mean Score"] = mean_rouge
result["Few-Shots"] = num_few_shots
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"INVALSI_{model_name}_top-p_{top_p}_temp_{temp}_token_{n_token_out}_{formatted_time}.json"
with open(file_name, "w") as outfile:
    json.dump(dict(result), outfile, ensure_ascii=False)

print("=" * shutil.get_terminal_size().columns)
