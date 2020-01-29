import json

def preprocess(dataset_path):
    # getting train data and dev data into csv file
    with open(dataset_path) as f:
        dataset = json.load(f)['data']

    # this snippet of code is for data extraction from json file.
    contexts = []
    questions = []
    answers_text = []
    answers_start = []
    titles = []
    for i in range(len(dataset)):
        topic = dataset[i]['paragraphs']
        title_ = dataset[i]['title']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                if len(q_a['answers']) > 0:
                    answers_start.append(q_a['answers'][0]['answer_start'])
                    answers_text.append(q_a['answers'][0]['text'])
                else:
                    answers_start.append(None)
                    answers_text.append(None)
                contexts.append(sub_para['context'])
                titles.append(title_)
    return contexts, questions, answers_start, answers_text, titles

