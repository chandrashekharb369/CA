import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Emoji unicode ranges
emoji_pattern = re.compile(
    '['
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F1E0-\U0001F1FF'
    '\U00002702-\U000027B0'
    '\U000024C2-\U0001F251'
    '\U0001FA70-\U0001FAFF'
    '\U00002500-\U00002BEF'
    '\U00002300-\U000023FF'
    '\u2600-\u26FF'
    '\u2700-\u27BF'
    ']+',
    flags=re.UNICODE
)

new_text = emoji_pattern.sub('', text)

# Put a generic non-emoji character or empty for page icon
new_text = new_text.replace('page_icon=\"\"', 'page_icon=\"CA\"')

# Add missing emojis if any from string concatenations that look weird like `ins['icon'] = ''`
# Actually it just strips them from the strings so "🏦 CA Suite" becomes " CA Suite"

if new_text != text:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Emojis removed from app.py')
else:
    print('No emojis found or regex failed')
