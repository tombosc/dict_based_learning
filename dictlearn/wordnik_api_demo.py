from wordnik import swagger, WordApi, AccountApi

client = swagger.ApiClient(
    'dd3d32ae6b4709e1150040139c308fb77446e0a8ecc93db31',
    'https://api.wordnik.com/v4')
word_api = WordApi.WordApi(client)

words = ['paint', 'mimic', 'mimics', 'francie', 'frolic', 'funhouse']
for word in words:
    print('=== {} ==='.format(word))
    defs = word_api.getDefinitions(word)
    if not defs:
        print("no definitions")
        continue
    for def_ in defs:
        fmt_str = "{} --- {}"
        print(fmt_str.format(def_.sourceDictionary, def_.text.encode('utf-8')))

account_api = AccountApi.AccountApi(client)
for i in range(5):
    print("Attempt {}".format(i))
    status = account_api.getApiTokenStatus()
    print("Remaining_calls: {}".format(status.remainingCalls))
