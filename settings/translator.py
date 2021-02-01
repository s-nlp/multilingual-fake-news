class Translator:

    def translate(self, request, dest):
        from requests import get
        import json

        response = get('https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl={}&q={}'.format(dest, request))
        trans_sentences = json.loads(response.text)['sentences']
        result = [sentence['trans'].strip() for sentence in trans_sentences if 'trans' in sentence.keys()]

        return '. '.join(result)


translator = Translator()
