from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2-large', device=0)


def generate(input_, content_blocks):
  length = len(input_) // 2 + 30
  content = 'I need to tell that' + ';'.join(content_blocks) + '. '
  result = generator(f'You asked me: "{input_}". I replied: "', max_length=length, num_return_sequences=1,
          return_full_text=False, prefix=content, temperature=None, num_beams=3, early_stopping=True,
          no_repeat_ngram_size=2)
  text = result[0]['generated_text']
  i = text.find('"')
  if i < 0:
    return text
  return text[:i]

generate('What do you think about me?', ['i like him so much', 'he is a disgusting person'])
