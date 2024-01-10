demonstrations = {
    'Spark': 'Log: Block broadcast_0 stored as values in memory (estimated size 384.0 B, free 317.5 KB)\nLogging statement: logger("Block {} stored as values in memory (estimated size {}, free {})", "broadcast_0", "384.0 B", "317.5 KB")\nLog: ',
    'Hadoop': 'Log: attempt_1445144423722_0020_m_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED\nLogging statement: logger("{} TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_1445144423722_0020_m_000000_0")\nLog: ',
    'Linux': 'Log: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=d211-116-254-214.rev.krline.net\nLogging statement: logger("authentication failure; logname= uid={} euid={} tty={} ruser= rhost={}", "0", "0", "NODEVssh", "d211-116-254-214.rev.krline.net")\nLog: ',
    'Zookeeper': 'Log: Received connection request /10.10.34.11:45307\nLogging statement: logger("Received connection request {}:{}", "/10.10.34.11", "45307")\nLog: ',
    'Android': 'Log: setSystemUiVisibility vis=40000500 mask=ffffffff oldVal=508 newVal=40000500 diff=40000008 fullscreenStackVis=0 dockedStackVis=0, fullscreenStackBounds=Rect(0, 0 - 720, 1280), dockedStackBounds=Rect(0, 0 - 0, 0)\nLogging statement: logger("setSystemUiVisibility vis=<*> mask=<*> oldVal=<*> newVal=<*> diff=<*> fullscreenStackVis=<*> dockedStackVis=<*>, fullscreenStackBounds=Rect(<*>, <*> - <*>, <*>), dockedStackBounds=Rect(<*>, <*> - <*>, <*>)", "40000500", "ffffffff", "508", "40000500", "40000008", "0", "0", "0", "0", "720", "1280", "0", "0", "0", "0")\nLog: ',
    'HDFS': 'Log: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864\nLogging statement: logger("BLOCK* NameSystem.addStoredBlock: blockMap updated: {} is added to {} size {}", "10.251.73.220:50010", "blk_7128370237687728475", "67108864")\nLog: ',
    'OpenSSH': 'Log: Connection closed by 212.47.254.145 [preauth]\nLogging statement: logger("Connection closed by {} [preauth]", "212.47.254.145")\nLog: ',
    'Proxifier': 'Log: chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS\nLogging statement: logger("{} open through proxy {} HTTPS", "proxy.cse.cuhk.edu.hk:5070", "proxy.cse.cuhk.edu.hk:5070")\nLog: ',
    'BGL': 'Log: generating core.2275\nLogging statement: logger("generating core.{}", "2275")\nLog: ',
    'HealthApp': 'Log: onExtend:1514038530000 14 0 4\nLogging statement: logger("onExtend:<*> <*> <*> <*>", "1514038530000", "14", "0", "4")\nLog: ',
    'OpenStack': 'Log: 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1910 time: 0.2808621\nLogging statement: logger("{} "GET {}" status: {} len: {} time: {}", "10.11.10.1", "/v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1", "200", "1910", "0")\nLog: ',
    'HPC': 'Log: ambient=28\nLogging statement: logger("ambient={}", "1514038530000", "28")\nLog: ',
    'Mac': '''Log: Cocoa scripting error for '0x00660011': four character codes must be four characters long.\nLogging statement: logger("Cocoa scripting error for '{}': four character codes must be four characters long.", "0x00660011")\nLog: ''',
    'Windows': 'Log: Session: 30546173_4261722401 initialized by client WindowsUpdateAgent.\nLogging statement: logger("Session: {} initialized by client WindowsUpdateAgent.", "30546173_4261722401")\nLog: ',
    'Apache': 'Log: jk2_init() Found child 6725 in scoreboard slot 10\nLogging statement: logger("jk2_init() Found child {} in scoreboard slot {}", "6725", "10")\nLog: ',
    'Thunderbird': 'Log: data_thread() got not answer from any [Thunderbird_A8] datasource\nLogging statement: logger("data_thread() got not answer from any [{}] datasource", "1514038530000")\nLog: ',
}

demonstrations_lograw = {
    'Spark': 'Log: 17/06/09 20:10:48 INFO storage.MemoryStore: Block broadcast_0 stored as values in memory (estimated size 384.0 B, free 317.5 KB)\nLogging statement: logger.info("Block {} stored as values in memory (estimated size {}, free {})", "broadcast_0", "384.0 B", "317.5 KB")\nLog: ',
    'Hadoop': 'Log: 2015-10-18 18:01:53,885 INFO [AsyncDispatcher event handler] org.apache.hadoop.mapreduce.v2.app.job.impl.TaskAttemptImpl: attempt_1445144423722_0020_m_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED\nLogging statement: logger.info("{} TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_1445144423722_0020_m_000000_0")\nLog: ',
    'Linux': 'Log: Jun 15 20:05:31 combo sshd(pam_unix)[24141]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=d211-116-254-214.rev.krline.net\nLogging statement: logger.info("authentication failure; logname= uid={} euid={} tty=NODEVssh ruser= rhost={}", "0", "0", "d211-116-254-214.rev.krline.net")\nLog: ',
    'Zookeeper': 'Log: 2015-07-29 19:04:12,394 - INFO  [/10.10.34.11:3888:QuorumCnxManager$Listener@493] - Received connection request /10.10.34.11:45307\nLogging statement: logger.info("Received connection request {}:{}", "/10.10.34.11", "45307")\nLog',
    'Android': 'Log: 03-17 16:13:38.954  2227  2227 I PhoneStatusBar: setSystemUiVisibility vis=40000500 mask=ffffffff oldVal=508 newVal=40000500 diff=40000008 fullscreenStackVis=0 dockedStackVis=0, fullscreenStackBounds=Rect(0, 0 - 720, 1280), dockedStackBounds=Rect(0, 0 - 0, 0)\nLogging statement: logger.info("setSystemUiVisibility vis=<*> mask=<*> oldVal=<*> newVal=<*> diff=<*> fullscreenStackVis=<*> dockedStackVis=<*>, fullscreenStackBounds=Rect(<*>, <*> - <*>, <*>), dockedStackBounds=Rect(<*>, <*> - <*>, <*>)", "40000500", "ffffffff", "508", "40000500", "40000008", "0", "0", "0", "0", "720", "1280", "0", "0", "0", "0")\nLog: ',
    'HDFS': 'Log: 081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864\nLogging statement: logger.info("BLOCK* NameSystem.addStoredBlock: blockMap updated: {} is added to {} size {}", "10.251.73.220:50010", "blk_7128370237687728475", "67108864")\nLog: ',
    'OpenSSH': 'Log: Dec 10 07:07:45 LabSZ sshd[24206]: Received disconnect from 52.80.34.196: 11: Bye Bye [preauth]\nLogging statement: logger.info("Received disconnect from {}: {}: Bye Bye [preauth]", "52.80.34.196", "11")\nLog: ',
    'Proxifier': 'Log: [10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS\nLogging statement: logger.info("{} open through proxy {} HTTPS", "proxy.cse.cuhk.edu.hk:5070", "proxy.cse.cuhk.edu.hk:5070")\nLog: ',
    'BGL': 'Log: - 1117955341 2005.06.05 R25-M0-N7-C:J02-U01 2005-06-05-00.09.01.903373 R25-M0-N7-C:J02-U01 RAS KERNEL INFO generating core.2275\nLogging statement: logger.info("generating core.{}", "2275")\nLog: ',
    'HealthApp': 'Log: 20171223-22:15:29:615|Step_LSC|30002312|onExtend:1514038530000 14 0 4\nLogging statement: logger.info("onExtend:<*> <*> <*> <*>", "1514038530000", "14", "0", "4")\nLog: ',
    'OpenStack': 'nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:10.978 25746 INFO nova.osapi_compute.wsgi.server [req-d81279b2-d9df-48b7-9c36-edab3801c067 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1910 time: 0.2808621\nLogging statement: logger.info("{} "GET {}" status: {} len: {} time: {}", "10.11.10.1", "/v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1", "200", "1910", "0")\nLog: ',
    'HPC': 'Log: 51338 node-3 node psu 1106496000 1 psu failure\ ambient=28\nLogging statement: logger.info("ambient={}", "1514038530000", "28")\nLog: ',
    'Mac': 'Log: Jul  4 23:22:09 calvisitor-10-105-162-105 Microsoft Word[14463]: Cocoa scripting error for \'0x00660011\': four character codes must be four characters long.\nLogging statement: logger.info("Cocoa scripting error for \'{}\': four character codes must be four characters long.", "0x00660011")\nLog: ',
    'Windows': 'Log: 2016-09-28 04:30:31, Info                  CBS    Session: 30546173_4261722401 initialized by client WindowsUpdateAgent.\nLogging statement: logger.info("Session: {} initialized by client WindowsUpdateAgent.", "30546173_4261722401")\nLog: ',
    'Apache': 'Log: [Sun Dec 04 04:51:08 2005] [notice] jk2_init() Found child 6725 in scoreboard slot 10\nLogging statement: logger.info("jk2_init() Found child {} in scoreboard slot {}", "6725", "10")\nLog: ',
    'Thunderbird': 'Log: - 1131566461 2005.11.09 tbird-admin1 Nov 9 12:01:01 local@tbird-admin1 /apps/x86_64/system/ganglia-3.0.1/sbin/gmetad[1682]: data_thread() got not answer from any [Thunderbird_A8] datasource\nLogging statement: logger.info("data_thread() got not answer from any [{}] datasource", "1514038530000")\nLog: ',
}

instruction_base = 'Given the following raw log, you should generate the corresponding logging statement (The logging statement is a single line of code that is used to log a message):\n'

instruction_lograw = '''Given the following raw log, you should generate the corresponding logging statement. The logging statement is a single line of code that is used to log a message. You need to understand the information in the raw log, determine the log level, extract all possible variables and replace them with placeholders, and finally generate the logging statement using the correct logging function and the log message with placeholders. All extracted variables should be passed as arguments to the logging function.'''

instruction_stage2 = '''Given the following logging statement (a single line of code used to log a message), generate the corresponding log template by identifying the variable parts and replacing them with placeholders in the form of <*>, only return the log template without function name or other information:
logging statement:{}
log template:'''


instruction_2stage = '''Given the following raw log, firstly you should generate the corresponding logging statement (The logging statement is a single line of code that is used to log a message), then you should generate the log template by identifying the variable parts and replacing them with placeholders in the form of <*>, return the code and log template without additional explainations:'''
# logging statement -> log template


def extract_and_replace(input_string):
    # 提取第一个“”中的内容
    start = input_string.find('"') + 1
    end = input_string.find('"', start)
    extracted = input_string[start:end]
    # 将所有{}替换成<*>
    replaced = extracted.replace('{}', '<*>')
    return replaced

# prosess the output of codellama


def transfer(input):
    lower_case_string = input.lower()
    log_index = lower_case_string.find('log')
    if log_index != -1:
        input = input[log_index:]
    return input[:input.find('\n')].strip()

# calculate the similarity


def calculate_similarity(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    # 计算两个集合的交集
    intersection = set1 & set2
    # 返回交集的长度，即两个列表中完全相等的元素的数量
    return len(intersection)

# 按EventTemplate列删除df中的重复行


def make_unique(df, subset):
    df_unique = df.drop_duplicates(subset=subset)
    return df_unique
