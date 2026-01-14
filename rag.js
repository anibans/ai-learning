// const fetch = require("node-fetch");
const cosineSimilarity = require("cosine-similarity");

const OLLAMA_BASE_URL = "http://localhost:11434";
const EMBED_MODEL = "nomic-embed-text";
const LLM_MODEL = "llama3";

const TOP_K = 3
const SIMILARITY_THRESHOLD = 0.3

const rawDocuments = [
  "Employees have a 30 day notice period. The notice period applies to all full-time employees.",
  "Office hours are from 10 AM to 6 PM. Employees are expected to be online during these hours.",
  "Health insurance is provided after 3 months. Coverage includes dependents."
]

const chunkText = (text, chunkSize=200, overlap=40)=>{
    const sentences = text.split(". ")
    const chunks = []

    const currentChunk = []
    let currentLength = 0

    for(const sentence of sentences){
        currentChunk.push(sentence)
        currentLength += sentence.length

        if(currentLength >= chunkSize){
            chunks.push(currentChunk.join(". "))
            currentChunk = currentChunk.slice(-Math.floor(overlap/50))
            currentLength = currentChunk.join(" ").length
        }
    }

    if(currentChunk.length > 0){
        chunks.push(currentChunk.join(". "))
    }

    return chunks
}

let chunks = []
let chunkId = 0

for(const doc of rawDocuments){
    const docChunks = chunkText(doc)
    for(const chunk of docChunks){
        chunks.push({
            id: ++chunkId,
            text: chunk
        })
    }
}

const embed = async text => {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
            model: EMBED_MODEL,
            prompt: text
        })
    })
    const data = await response.json()
    return data.embedding
}

const embedAllChunks = async () => {
    for(const chunk of chunks){
        chunk.embedding = await embed(chunk.text)
    }
}

const retrieve = async (query) => {
    const queryEmbedding = await embed(query)

    const scored = chunks.map(chunk => ({
        chunk,
        score: cosineSimilarity(queryEmbedding,chunk.embedding)
    }))

    return scored
    .filter(item => item.score >= SIMILARITY_THRESHOLD)
    .sort((a,b)=>b.score - a.score)
    .slice(0,TOP_K)
}

const generateAnswer = async (question,context) => {
    const prompt = `
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know."

        Context:
        ${context}


        Question:
        ${question}
    `

    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            model: LLM_MODEL,
            prompt,
            stream: false
        })
    })

    const data = await response.json()
    return data.response
}

(async () => {
    console.log('Embedding knowledge base...')
    await embedAllChunks()

    const question = "What is the notice period?"

    console.log("\nQuestion:", question)

    const results = await retrieve(question)

    if(results.length === 0){
        console.log("\nAnswer:\nI don't know")
        return
    }

    console.log("\nRetrieved Chunks:")
    results.forEach( result => 
        console.log(`- (${result.score.toFixed(2)}) ${result.chunk.text}`)
    )

    const context = results.map( result => result.chunk.text).join("\n---\n")

    const answer = await generateAnswer(question,context)

    console.log("\nAnswer:")
    console.log(answer)
})()