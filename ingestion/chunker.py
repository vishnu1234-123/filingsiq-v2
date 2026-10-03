import tiktoken

_ENCODING=tiktoken.get_encoding("cl100k_base")
MAX_TOKENS=500

def _count_tokens(text:str)->int:
    if not isinstance(text,str):
        print("BAD PIECE TYPE:", type(text), "VALUE:", text)
        raise TypeError("chunk piece is not a string")
    return len(_ENCODING.encode(text))

def _split_text_block(text:str,max_tokens:int)->list[str]:
    paragraphs=[p.strip() for p in text.split("\n") if p.strip()]
    chunks=[]
    current=""

    for para in paragraphs:
        candidate=(current+" "+para).strip() if current else para
        if _count_tokens(candidate)>max_tokens and current:
            chunks.append(current)
            current=para
        else:
            current=candidate
    if current:
        chunks.append(current)

    final=[]
    for c in chunks:
        if _count_tokens(c)<=max_tokens:
            final.append(c)
        else:
            tokens=_ENCODING.encode(c)
            for i in range(0,len(tokens),max_tokens):
                final.append(_ENCODING.decode(tokens[i:i+max_tokens]))
    return final


def chunk_sections(sections:list[dict],max_tokens:int=MAX_TOKENS)->list[dict]:
    chunks=[]

    for section in sections:
        heading=section["heading"]
        text_buffer=""
        for kind,content in section["blocks"]:
            if kind=="TABLE":
                if not content.strip():
                    continue
                if text_buffer.strip():
                    for piece in _split_text_block(text_buffer,max_tokens):
                        chunks.append({
                            "heading":heading,"type":"text",
                            "text":piece,"token_count":_count_tokens(piece),
                        })
                    text_buffer=""
                chunks.append({
                    "heading":heading,"type":"table",
                    "text":content,"token_count":_count_tokens(content),
                })
            else:
                text_buffer+="\n"+content
        if text_buffer.strip():
            for piece in _split_text_block(text_buffer,max_tokens):
                chunks.append({
                    "heading":heading,"type":"text",
                    "text":piece,"token_count":_count_tokens(piece),
                })
    return chunks


if __name__ == "__main__":
    import os
    from lxml import etree
    from section_splitter import split_into_sections

    filepath = "../data/raw/aapl-20250927.htm"
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        exit(1)

    tree = etree.parse(filepath, etree.HTMLParser())
    sections = split_into_sections(tree)
    chunks = chunk_sections(sections)

    print(f"total chunks: {len(chunks)}")
    table_chunks = [c for c in chunks if c["type"] == "table"]
    print(f"table chunks: {len(table_chunks)}")
    print("\nmax token count in any single chunk:", max(c["token_count"] for c in chunks))
    print("\nsample table chunk:")
    print(table_chunks[0]["heading"], "|", table_chunks[0]["text"][:200])
    biggest = max(chunks, key=lambda c: c["token_count"])
    print(biggest["type"], biggest["token_count"], biggest["heading"])
    text_chunks = [c for c in chunks if c["type"] == "text"]
    print("max text chunk tokens:", max(c["token_count"] for c in text_chunks))