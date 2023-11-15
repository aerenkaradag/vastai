def onlyHighest(Lines):
    newLines = []
    before = 0
    for line in Lines:
        line1 = line.split(',')
        if(line1[0] != before):
            if(float(line1[5]) > 0.5):
              newLines.append(line)
            else:
              newLines.append("{},0,0,0,0,0 \n".format(line1[0]))
            before = line1[0] 
    return newLines

def onlyHighestsim(Lines):
    newLines = []
    before = 0
    for line in Lines:
        line1 = line.split(',')
        if(line1[0] != before):
            if(float(line1[1])> 0):
              newLines.append(line)
            else:
              newLines.append("{},0\n".format(line1[0]))
            before = line1[0] 
    return newLines


def ma(number, Lines):
    if(len(Lines) > number):
      count = 0
      list1 = []
      average = []
      while(count < number):
        Lines1 = Lines[count].split(',')[5]
        list1.append(float(Lines1))
        count = count + 1
      average.append(sum(list1)/len(list1))
      while( count < len(Lines)):
          Lines1 = Lines[count].split(',')[5]
          list1.pop(0)
          list1.append(float(Lines1))
          average.append(sum(list1)/len(list1))
          count = count + 1
    else:
         average = []
    return average

def ma50(Lines):
    average = ma(50,Lines)
    return average
def ma25(Lines):
    average = ma(25,Lines)
    return average

def masim(number, Lines):
    if(len(Lines) > number):
      count = 0
      list1 = []
      average = []
      while(count < number):
        Lines1 = Lines[count].split(',')[1]
        list1.append(float(Lines1))
        count = count + 1
      average.append(sum(list1)/len(list1))
      while( count < len(Lines)):
          Lines1 = Lines[count].split(',')[1]
          list1.pop(0)
          list1.append(float(Lines1))
          average.append(sum(list1)/len(list1))
          count = count + 1
    else:
         average = []
    return average

def average(Lines):
    count = 0
    sum = 0
    av_1 = 0
    if len(Lines) == 0:
      return 0,0
    else:
      while count < len(Lines):
          sum = sum + float(Lines[count].split(',')[5])
          count = count + 1 
          if(count == len(Lines) - 1):
            av_1 = sum
      return sum / len(Lines), av_1 / (len(Lines) - 1)

def averagesim(Lines):
    count = 0
    sum = 0
    av_1 = 0
    if len(Lines) == 0:
      return 0,0
    else:
      while count < len(Lines):
          sum = sum + float(Lines[count].split(',')[1])
          count = count + 1 
          if(count == len(Lines) - 1):
            av_1 = sum
      return sum / len(Lines), av_1 / (len(Lines) - 1)