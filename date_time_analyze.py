import re

class DateTimeAnalyzer:
    def __init__(self):
        month_date = r"([0-9一二三四五六七八九十]*|[這上下]*個)月[0-9一二三四五六七八九十]+[號日]?" + r"|" + r"[0-9]+/[0-9]+"
        week_date = r"([這上下]*個?(禮拜|星期|周|週|明|後|昨|前)+[1-7一二三四五六七日天])"
        relative_date = r"[0-9一二兩三四五六七八九十]+(天|月|年|小時)之?[前後]?"

        self.date_analyze = re.compile(month_date + r"|" + week_date + r"|" + relative_date)
        self.time_analyze = re.compile(r"([上下]午)?[0-9一二三四五六七八九十]+[點時:：][0-9一二三四五六七八九十半整]*分?[以之]?[前後]?")
        
    def analyze_date(self, date):
        return self.date_analyze.search(date)

    def analyze_time(self, time):
        return self.time_analyze.search(time)


if __name__ == '__main__':
    def analyze(string):
        date = analyzer.analyze_date(string)
        time = analyzer.analyze_time(string)

        # print(date, time)

        if date != None:
            print("date:", date.group())
        else:
            print("date not found")

        if time != None:
            print("time:", time.group())
        else:
            print("time not found")
        print()

    analyzer = DateTimeAnalyzer()

    test_data = ["我這禮拜天晚上8點整可以",
        "好啊那就決定是2月10號，16：30了",
        "那個人說他比較想要下下下下禮拜二欸",
        "要九點半以後才可以來討論",
        "投票結果：2/29下午4點半",
        "王大明說他上個禮拜星期四下午七點二十二分有去吃麥當勞"]

    for i in test_data:
        print("輸入分析字串：", i)
        analyze(i)

    while True:
        text = input("輸入分析字串： ")
        analyze(text)
