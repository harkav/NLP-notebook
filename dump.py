def most_similar(input_document: str, corpus: list[str]) -> tuple[str, float]:
    all_words, term_index_dict, idf_vector = create_tf_idf_context(corpus)

    corpus_document_objects = []

    for doc in corpus:
        corpus_document_objects.append(
            create_object_new(doc, all_words, term_index_dict, idf_vector)
        )

    input_doc = create_object_new(
        input_document, all_words, term_index_dict, idf_vector
    )

    highest = ("", -2)

    for corp_doc in corpus_document_objects:
        similarity = cosine_similarity(corp_doc.vector, input_doc.vector)
        print(similarity)
        print(corp_doc.content)
        if similarity > highest[1]:
            highest = (corp_doc.content, similarity)

    return highest