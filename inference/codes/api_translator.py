from urllib import parse, request


class PapagoTranslator:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.url = 'https://naveropenapi.apigw.ntruss.com/nmt/v1/translation'

    def translate(self, text, src_lang='en', tgt_lang='ko'):
        encoded_text = parse.quote(text)
        data = f'source={src_lang}&target={tgt_lang}&text={encoded_text}'
        
        trans_request = request.Request(self.url)
        trans_request.add_header('X-NCP-APIGW-API-KEY-ID', self.client_id)
        trans_request.add_header('X-NCP-APIGW-API-KEY', self.client_secret)

        trans_response = request.urlopen(trans_request, data=data.encode('utf-8'))
        responded_code = trans_response.getcode()

        if responded_code == 200:
            responded_body = trans_response.read()
            translation = responded_body.decode('utf-8')
            translation = eval(translation)['message']['result']['translatedText']
            return translation
        else:
            raise Exception(f"HTTPError: {responded_code}")