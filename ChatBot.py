import os
import fitz
import pytesseract
import tempfile
from PIL import Image
from pdf2image import convert_from_path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# ---------------------------
# 1. Embedding Function
# ---------------------------
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") 

# ---------------------------
# 2. Text Extraction from PDF (with fallback OCR)
# ---------------------------

def extract_text_from_pdf(pdf_path, footer_margin=50, header_margin=50):
    import fitz
    import pytesseract
    from PIL import Image
    import io

    def column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True):
        """Determine bboxes which wrap a column."""
        paths = page.get_drawings()
        bboxes = [] 

        path_rects = []

        img_bboxes = []

        vert_bboxes = []

        clip = +page.rect
        clip.y1 -= footer_margin  
        clip.y0 += header_margin


        def can_extend(temp, bb, bboxlist):
            """Determines whether rectangle 'temp' can be extended by 'bb'
            without intersecting any of the rectangles contained in 'bboxlist'.

            Items of bboxlist may be None if they have been removed.

            Returns:
                True if 'temp' has no intersections with items of 'bboxlist'.
            """
            for b in bboxlist:
                if not intersects_bboxes(temp, vert_bboxes) and (
                    b == None or b == bb or (temp & b).is_empty
                ):
                    continue
                return False

            return True

        def in_bbox(bb, bboxes):
            """Return 1-based number if a bbox contains bb, else return 0."""
            for i, bbox in enumerate(bboxes):
                if bb in bbox:
                    return i + 1
            return 0

        def intersects_bboxes(bb, bboxes):
            """Return True if a bbox intersects bb, else return False."""
            for bbox in bboxes:
                if not (bb & bbox).is_empty:
                    return True
            return False

        def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
            """Extend a bbox to the right page border.

            Whenever there is no text to the right of a bbox, enlarge it up
            to the right page border.

            Args:
                bboxes: (list[IRect]) bboxes to check
                width: (int) page width
                path_bboxes: (list[IRect]) bboxes with a background color
                vert_bboxes: (list[IRect]) bboxes with vertical text
                img_bboxes: (list[IRect]) bboxes of images
            Returns:
                Potentially modified bboxes.
            """
            for i, bb in enumerate(bboxes):
                
                if in_bbox(bb, path_bboxes):
                    continue

                if in_bbox(bb, img_bboxes):
                    continue

                temp = +bb
                temp.x1 = width

      
                if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                    continue

                check = can_extend(temp, bb, bboxes)
                if check:
                    bboxes[i] = temp 

            return [b for b in bboxes if b != None]

        def clean_nblocks(nblocks):
            """Do some elementary cleaning."""

            blen = len(nblocks)
            if blen < 2:
                return nblocks
            start = blen - 1
            for i in range(start, -1, -1):
                bb1 = nblocks[i]
                bb0 = nblocks[i - 1]
                if bb0 == bb1:
                    del nblocks[i]

            y1 = nblocks[0].y1
            i0 = 0
            i1 = -1 

            for i in range(1, len(nblocks)):
                b1 = nblocks[i]
                if abs(b1.y1 - y1) > 10:
                    if i1 > i0:
                        nblocks[i0 : i1 + 1] = sorted(
                            nblocks[i0 : i1 + 1], key=lambda b: b.x0
                        )
                    y1 = b1.y1
                    i0 = i
                i1 = i
            if i1 > i0:
                nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
            return nblocks

        for p in paths:
            path_rects.append(p["rect"].irect)
        path_bboxes = path_rects

        path_bboxes.sort(key=lambda b: (b.y0, b.x0))

        for item in page.get_images():
            img_bboxes.extend(page.get_image_rects(item[0]))

        blocks = page.get_text(
            "dict",
            flags=fitz.TEXTFLAGS_TEXT,
            clip=clip,
        )["blocks"]

        for b in blocks:
            bbox = fitz.IRect(b["bbox"])

            if no_image_text and in_bbox(bbox, img_bboxes):
                continue

            line0 = b["lines"][0] 
            if line0["dir"] != (1, 0): 
                vert_bboxes.append(bbox)
                continue

            srect = fitz.EMPTY_IRECT()
            for line in b["lines"]:
                lbbox = fitz.IRect(line["bbox"])
                text = "".join([s["text"].strip() for s in line["spans"]])
                if len(text) > 1:
                    srect |= lbbox
            bbox = +srect

            if not bbox.is_empty:
                bboxes.append(bbox)

        bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

        bboxes = extend_right(
            bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes
        )

        if bboxes == []:
            return []

        nblocks = [bboxes[0]]
        bboxes = bboxes[1:]

        for i, bb in enumerate(bboxes):
            check = False

            for j in range(len(nblocks)):
                nbb = nblocks[j]

                if bb == None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                    continue

                if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                    continue

                temp = bb | nbb
                check = can_extend(temp, nbb, nblocks)
                if check == True:
                    break

            if not check:
                nblocks.append(bb)
                j = len(nblocks) - 1
                temp = nblocks[j]

            check = can_extend(temp, bb, bboxes)
            if check == False:
                nblocks.append(bb)
            else:
                nblocks[j] = temp
            bboxes[i] = None

        nblocks = clean_nblocks(nblocks)
        return nblocks



    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        bboxes = column_boxes(page, footer_margin=footer_margin, header_margin=header_margin)
        if bboxes:
            bboxes = sorted(bboxes, key=lambda b: (round(b.y0, 1), round(b.x0, 1)))
            for bbox in bboxes:
                words = page.get_text("words", clip=bbox)
                words.sort(key=lambda w: (round(w[1], 1), round(w[0], 1)))  # (y0, x0)
                text = " ".join(w[4] for w in words if w[4].strip())
                if text.strip():
                    all_text.append(text)

        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    all_text.append(ocr_text)
            except Exception as e:
                print(f"Error processing image {image_index}: {e}")

    return "\n".join(all_text)



# ---------------------------
# 3. Vector Store (Create + Persist)
# ---------------------------
def create_vector_store(pdf_paths, persist_directory="vector_store"):
    all_text = ""
    for path in pdf_paths:
        all_text += extract_text_from_pdf(path) + "\n"

    print(f"Extracted {len(all_text)} characters from {len(pdf_paths)} PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_text(all_text)

    if not chunks:
        print("No text chunks generated. Cannot create Chroma DB.")
        return None

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    print(f"Indexed {len(chunks)} chunks into Chroma DB.")
    return vectordb


# ---------------------------
# 4. Query Vector Store
# ---------------------------
def query_vector_store(question, persist_directory="vector_store", k=5):
    if not os.path.exists(persist_directory):
        print(f"Vector store not found at {persist_directory}. Please create it first.")
        return []

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    docs = vectordb.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]


# ---------------------------
# 5. Prompt Construction
# ---------------------------
def build_prompt(question, top_chunks):
    context = "\n\n".join([f"Excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
    return f"""
You are an assistant answering questions based strictly on the following document excerpts:

{context}

Question: {question}
Answer concisely. If the answer is not in the excerpts, reply with "Not found in the document."
""".strip()


# ---------------------------
# 6. Ask Qwen (requires pre-defined `client`)
# ---------------------------
def ask_qwen(prompt):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        response_text = response.choices[0].message.content.strip()
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        response_text = re.split(r"<think>", response_text)[0].strip()
        response_text = re.sub(r'\n{2,}', '\n', response_text).strip()
        return response_text
    except Exception as e:
        return f"Error calling Qwen API: {e}"


# ---------------------------
# 7. RAG Pipeline
# ---------------------------
def answer_question_rag(pdf_paths, question):
    persist_directory = "vector_store"
    if not os.path.exists(persist_directory):
        print("Creating vector store...")
        vectordb = create_vector_store(pdf_paths, persist_directory)
        if vectordb is None: 
            print("Failed to create vector store. Cannot answer question.")
            return "Error: Could not process the PDF."

    print("Querying vector store...")
    top_chunks = query_vector_store(question, persist_directory)

    if not top_chunks:
        print("No relevant chunks found in the vector store.")
        return "Not found in the document."


    print("\nTop Chunks Selected:")
    for i, chunk in enumerate(top_chunks):
        print(f"Chunk {i+1}:\n{chunk[:200]}...\n")

    print("Asking Qwen...")
    prompt = build_prompt(question, top_chunks)
    answer = ask_qwen(prompt)

    print("\nFinal Answer:\n", answer)
    return answer


# ---------------------------
# 8. Main Entry Point
# ---------------------------
if __name__ == "__main__":
  while True:
      pdf_paths = [
          "/content/education-large.pdf",
          "/content/what-is-history.pdf",
          "/content/paper02.pdf",
          "/content/ocrim.pdf"
      ]
      question = input("Ask a question: ")
      if question.lower() == "exit":
          break
      answer_question_rag(pdf_paths, question)