import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('eval/result2022-b2ebc7ea7b49.json', scope)
gc = gspread.authorize(credentials)

sheet = gc.open('eval_result2022').sheet1

def send_gs(gs_result_list):
    row = gs_result_list
    index = 2
    sheet.insert_row(row, index)

