import csv
import numpy as np
classidfile = './data/classid.csv' 

def from_666_to_140(filename,list1):
    csvfile=open(filename,'r')
    reader=csv.reader(csvfile)
    idfile=open(classidfile,'r')
    idreader=csv.reader(idfile)
    classid_total=[]
    for item in idreader:
        classid_total.append([item,0])
    
    classid_total=classid_total[1:]
    b_result=[]
    for item in reader:
        b_result.append(item[0][:6])
    b_result=b_result[1:]
#    print b_result
    list2=[]
    for i in range(len(list1)):
        list2.append([b_result[i],list1[i]])
#    print list2
    for i in range(len(classid_total)):
        for j in range(len(list2)):
#            print str(classid_total[i][0])[2:-2],list2[j][0]
            if str(classid_total[i][0])[2:-2]==list2[j][0]:
                classid_total[i][1]=classid_total[i][1]+list2[j][1]
#    print classid_total
    total=0
    list3=[]
    for item in classid_total:
        total=total+item[1]
        list3.append(item[1])
#    print list3
#    print total
#    print np.sum(list3)
    return  list3
#    data = genfromtxt(filename, delimiter=',', skip_header=1)
#    data = data[:,0]
#    for i in range(len(data)):
#        print str(float(data[i]))[:7]
    
def to_csv(file_name,list_r):
    import time    
    now = time.strftime('-%m-%d-%H-%M')
    fullname=file_name+now+'.csv'
    csvfile=file(fullname,'wb')
    writer=csv.writer(csvfile)
    writer.writerow(['predict_date','predict_quantity'])
    for i in range(len(list_r)):
        data=[201711,float(list_r[i])]
        writer.writerow(data)

    csvfile.close()