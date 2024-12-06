from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.youtube import TranscriptFormat
from yt_dlp import YoutubeDL  

def get_youtube_title(youtube_url):
    """
    Fetch the YouTube video title using yt-dlp.
    """
    try:
        with YoutubeDL() as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Unknown Title")
    except Exception as e:
        print(f"Error fetching title with yt-dlp: {e}")
        return "Unknown Title"

def load_youtube_texts(youtube_url, chunk_size=100, overlap=20):
    try:
        print(f"Processing YouTube video from {youtube_url}")
        loader = YoutubeLoader.from_youtube_url(youtube_url) #video transcripts-YoutubeLoader
        documents = loader.load()
        print(f"Processed YouTube video: {youtube_url}")

        text = []
        for doc in documents:
            title = get_youtube_title(youtube_url)
            text.append(f"{title}\n{doc.page_content}")

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        
        chunks = []
        for doc in text:
            chunks.extend(splitter.split_text(doc))  

        print(f"Generated {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error processing YouTube video {youtube_url}: {e}")
        return []