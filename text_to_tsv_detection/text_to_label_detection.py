import csv
import re
import string
names={
    "Happy kitty 子馨:D", "091吳宣",
    "Lydia Huang","吳永璿","李維哲","庭柔","林冠宇Kenlin","林潔禧","Kaladin Tseng","杰",#dia8
    "宇晨","陳宣佑™","1-33朱易頡","(Enderless)Ender Hsu",#dia7
    "振宇","陳星宏",
    "朱建宇","顯哲","葉俊廷","黃浩瑋"#dia0123


}
for i in range(0 ,10):
    try:
        f = open(".\dataset\dia"+str(i)+".txt", "r",encoding="utf-8")
        
        Lines = f.readlines()
        f.close()
        date=""
        time=""
        holder=""
        context=""
        concat_lines=["red", "blue", "green"]
        concat_lines.clear()
        for line in Lines:
            line=line.rstrip("\n")
            if len(line)>=14 and line[4]=='.'and line[7]=='.':#date
                concat_lines.append(line)
                continue
                
            if  len(line)>=5 and line[2]==':':
                concat_lines.append(line)#message
                continue
            concat_lines[-1]+=line#part of another message

            
        
        subNum=0
        start_flag=0
        for line in concat_lines:
            if(line.find("-start")!=-1):
                out_file=open(".\label_detection\dia"+str(i)+'_'+str(subNum)+'_label_detection.tsv', 'wt',encoding="utf-8")
                tsv_writer = csv.writer(out_file,delimiter='\t', lineterminator='\n' )
                tsv_writer.writerow(['label(0-irrelevant, 1-time, 2-location)', 'holder','date','context'])
                start_flag=1
                continue
            elif(line.find("-end")!=-1):
                out_file.close()
                subNum+=1
                start_flag=0
                continue
            if len(line)>=14 and line[4]=='.'and line[7]=='.':#date
                date=line
                continue
                
            if  len(line)>=5 and line[2]==':' and start_flag==1:
                time=line[0:5]
                count=0
                for n in names:
                    flag=0
                    count+=1
                    if len(n)==1 and line[6]==n[0]:
                        flag=1
                    elif len(n)==2 and line[6]==n[0] and line[7]==n[1]:
                        flag=1
                    elif len(n)>=3 and line[6]==n[0] and line[7]==n[1] and line[8]==n[2]:
                        flag=1
                    if flag==1:
                        holder=n
                        if len(n)>=2:
                            holder=holder[0]+"__"+str(count)
                        context=line[(6+len(n)):]
                        break
                count=0
                for n in names:
                    count+=1
                    if context.find(n)!=-1:
                        context=context.replace(n,n[0]+"_"+str(count))
                tsv_writer.writerow(['0', holder,date+time,context])    
                   
                
                
                
                    
        
    except IOError:
        print("File"+str(i)+" not accessible")

    
    