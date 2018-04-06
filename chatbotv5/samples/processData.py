import sqlite3

def get_db():
    conn = sqlite3.connect('wechat.db')
    ret = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(ret.fetchall())
    result = conn.execute("select datetime(subStr(cast(m.createTime as text),1,10),'unixepoch', 'localtime') as theTime,case m.isSend when 0 then r.nickname when 1 then '我'end as person,m.content from message m inner join rcontact r on m.talker = r.username where m.type=1 and r.nickname = 'Andrea21'")
    return result



formedAn = []
formedQe = []
json_format = False
result = get_db()
last = '我'
temp_formedAn = []
temp_formedQe = []
count = 0
length = 0
# for row in result:
#     length += 1
#     print(length)
for row in result:
    count += 1
    if last == '我' and row[1] == 'Andrea21':
        if temp_formedAn and not temp_formedQe:
            temp_formedAn = []
            temp_formedQe = []
            # 针对第一句是回答的情况
        for que in temp_formedQe:
            for ans in temp_formedAn:
                formedQe.append(que)
                formedAn.append(ans)
                # print(que)
                print(ans)
        temp_formedAn = []
        temp_formedQe = []
        temp_formedQe.append(row[2])
    if row[1] == '我':
        temp_formedAn.append(row[2])
    # if last == 'Andrea21' and row[1] == '我':
    #     temp_formedAn.append(row[2])
    if last == 'Andrea21' and row[1] == 'Andrea21':
        temp_formedQe.append(row[2])
    last = row[1]
    print(count)
    print(count)
    print(count)
    print(count)
    print(count)
    print(count)
    print(count)
    print(count)
    print(count)
    # print(length)
    # print(length)
    # print(length)


wrSTR = ''
for each in formedQe:
    # print(each)
    wrSTR += str(each)+'\n'



file = open('question',"w", encoding='utf-8')
file.write(wrSTR)

# 关闭
file.close()


wrSTR = ''
for each in formedAn:
    # print(each)
    wrSTR += str(each)+'\n'

print(wrSTR)
file = open('answer',"w", encoding='utf-8')

file.write(wrSTR)
# 关闭
file.close()

# re.split(r'[ \t\n]+', raw)

#{'我': '她肯定不好意思说啊'}\n{'Andrea21':
#{'我':
#{'我':
# ['./samples/question', './samples/answer']
