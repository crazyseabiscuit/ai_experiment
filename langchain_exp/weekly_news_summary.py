from html.parser import HTMLParser
import re
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return ' '.join(self.text)


def fetch_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        parser = MyHTMLParser()
        parser.feed(response.text)
        text = parser.get_text()
        return text
    else:
        raise Exception(f'Failed to fetch content from {url}')


def clean_text(text):
    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def score_sentences(sentences):
    scores = {}
    for index, sentence in enumerate(sentences):
        # Score based on sentence length and position
        score = len(sentence.split()) + (1 / (index + 1))
        scores[index] = score
    return scores


def summarize_text(text, num_sentences=3):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\\.)(?<=\\.\\s)', text)
    sentences = [clean_text(sentence) for sentence in sentences if clean_text(sentence)]
    scores = score_sentences(sentences)
    top_sentences_indices = heapq.nlargest(num_sentences, scores, key=scores.get)
    summary = ' '.join([sentences[i] for i in sorted(top_sentences_indices)])
    return summary


def generate_knowledge_graph(summary):
    words = summary.split()
    edges = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def display_knowledge_graph(graph):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', font_size=10, node_size=1000)
    plt.show()


def send_email(subject, body, image_path, to_emails):
    from_email = 'your_email@example.com'
    password = 'your_password'
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = ', '.join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-ID', '<image1>')
        msg.attach(img)
    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_emails, text)
    server.quit()


def main():
    url = input('Enter the website URL: ')
    to_emails = input('Enter recipient email addresses separated by commas: ').split(', ')
    content = fetch_content(url)
    summary = summarize_text(content)
    graph = generate_knowledge_graph(summary)
    display_knowledge_graph(graph)
    plt.savefig('knowledge_graph.png')
    send_email('Webpage Summary', summary, 'knowledge_graph.png', to_emails)
    print('Summary and knowledge graph sent successfully.')


if __name__ == '__main__':
    main()
