
wrSTR = ''
with open('./question', 'r') as question_file:
      with open('./answer', 'r') as answer_file:
          while True:
              question = question_file.readline()
              answer = answer_file.readline()
              if question and answer:
                  question = question.strip()
                  answer = answer.strip()
                  wrSTR += str(question)+'\n'
                  wrSTR += str(answer)+'\n'
              else:
                  break
file = open('./corpus.raw',"w", encoding='utf-8')
file.write(wrSTR)
file.close()
