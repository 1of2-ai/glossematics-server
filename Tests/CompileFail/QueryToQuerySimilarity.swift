import gloss_server

func invalidSimilarity(query: QueryEmbedding) throws -> Double {
    try query.similarity(to: query)
}
