# Selected student implementations from the Fall 2022 submission.
# Imported in 2026; fragments use the original notebook globals.

# Student implementation, submitted cell 25
def make_grams(tweets, n=2):
  grams =  list(ngrams(tweets,n))
  return grams

# Student implementation, submitted cell 25
def calculate_counts(grams):
  counts = Counter(grams)
  del counts[(vocabulary['</s>'],vocabulary['<s>'])]
  return counts

# Student implementation, submitted cell 32
def get_next_word(next_word_probs, n, start_text):
  first_word = vocabulary[start_text[-1]]
  something = []
  next = -1

  for i in start_text:
    something.append(vocabulary[i])

  new_words = []
  for i in range(n):
    max = 0
    maxW= []
    loop = 0
    for every_grams, every_count in next_word_probs.items():
      if(loop == len(next_word_probs)-1):
        break
      if(first_word == every_grams[0] and every_count > max):
        max = every_count
        maxW.append(every_grams[1])
      loop +=1
    if(max != 0):
      next = random.choice(maxW[-3:])
    if(next == -1):
      next = vocabulary['bad']

    if(next != vocabulary['</s>'] and next != vocabulary['<s>']):
      something.append(next)
    first_word = next
  return something

# Student implementation, submitted cell 37
def t_get_next_word(next_word_probs,biprobs, n, start_text):
  something = []
  for i in start_text:
    something.append(vocabulary[i])
  first_word = (start_text[-2], start_text[-1])
  new_words = []
  for i in range(n):
    next = -1
    maxW= []
    max = 0
    loop = 0
    for every_grams, every_count in next_word_probs.items():
      if(loop == len(next_word_probs)-1):
        break
      if((first_word[0],first_word[1]) == (every_grams[0],every_grams[1]) and every_count > max):
        max = every_count
        maxW.append(every_grams[2])
      loop +=1
      if(max != 0):
        next = random.choice(maxW[-3:])
    if(next == -1):
      b_first_world = first_word[1]
      max = 0
      loop = 0
      for every_grams, every_count in biprobs.items():
        if(loop == len(biprobs)-1):
          break
        if(b_first_world == every_grams[0] and every_count > max):
          max = every_count
          maxW.append(every_grams[1])
        loop +=1
        if(max != 0):
          next = random.choice(maxW[-3:])
    if(next == -1):
      next = vocabulary['bad']

    if(next != vocabulary['</s>'] and next != vocabulary['<s>']):
      something.append(next)
    temp = first_word
    first_word = (temp[1], next)
  return something
