require("dotenv").config();
const axios = require("axios");
const fs = require("fs");
const OpenAI = require("openai");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const Groq = require("groq-sdk");

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const { CANDIDATE_1, CANDIDATE_2 } = require("./prompts/candidates");
const CONTEXT = require("./prompts/context");
const questions = require("./prompts/questions");

function printFormatted(type, content) {
    const separator = "----------------------------------------";
    if (type === "question") {
        console.log(`\n${separator}\nPergunta: ${content}\n${separator}`);
    } else if (type === "response") {
        console.log(`\nResposta de ${content.candidate}: ${content.response}`);
    } else if (type === "rebuttal") {
        console.log(`\nRéplica de ${content.candidate}: ${content.rebuttal}`);
    } else if (type === "moderator") {
        console.log(`\nModerador (Gemini): ${content}\n`);
    }
}

async function askOpenai(prompt, model) {
    try {
        const completion = await openai.chat.completions.create({
            messages: [
                { role: "system", content: CONTEXT },
                { role: "user", content: prompt },
            ],
            model: model,
        });
        return completion.choices[0].message.content;
    } catch (error) {
        console.error(
            `Error asking assistant with model ${JSON.stringify(model)}: ${
                error.response
                    ? JSON.stringify(error.response.data, null, 2)
                    : error.message
            }`
        );
    }
}

async function askGroq(prompt, model) {
    try {
        const response = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: model,
        });
        const content = response.choices[0]?.message?.content || "";
        console.log(`Groq response content: ${content}`);
        return content;
    } catch (error) {
        console.error(
            `Error asking Groq: ${
                error.response
                    ? JSON.stringify(error.response.data, null, 2)
                    : error.message
            }`
        );
    }
}

async function askGemini(prompt) {
    try {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const result = await model.generateContent(prompt);
        return result.response.text();
    } catch (error) {
        console.error(
            `Error asking Gemini: ${
                error.response
                    ? JSON.stringify(error.response.data, null, 2)
                    : error.message
            }`
        );
    }
}

async function moderator(questions) {
    let responses = [];
    for (let i = 0; i < questions.length; i++) {
        let question = questions[i];
        let moderation = await askGemini(`Pergunta ${i + 1}: ${question}`);
        printFormatted("moderator", `Pergunta ${i + 1}: ${question}`);

        let response1, response2, rebuttal1, rebuttal2;
        if (i % 2 === 0) {
            response1 = await respond("ChatGPT 4o Mini", question, CANDIDATE_1);
            response2 = await respond("ChatGPT 3.5", question, CANDIDATE_2);

            rebuttal2 = await rebuttal(
                "ChatGPT 3.5",
                response1.response,
                CANDIDATE_2
            );
            rebuttal1 = await rebuttal(
                "ChatGPT 4o Mini",
                response2.response,
                CANDIDATE_1
            );
        } else {
            response1 = await respond("ChatGPT 3.5", question, CANDIDATE_2);
            response2 = await respond("ChatGPT 4o Mini", question, CANDIDATE_1);

            rebuttal2 = await rebuttal(
                "ChatGPT 4o Mini",
                response1.response,
                CANDIDATE_1
            );
            rebuttal1 = await rebuttal(
                "ChatGPT 3.5",
                response2.response,
                CANDIDATE_2
            );
        }
        responses.push({
            question: question,
            "ChatGPT 4o Mini": {
                response: response1.response,
                rebuttal: rebuttal1.rebuttal,
            },
            "ChatGPT 3.5": {
                response: response2.response,
                rebuttal: rebuttal2.rebuttal,
            },
        });
    }
    saveResponses(responses);
    return responses;
}

async function respond(candidate, question, model) {
    printFormatted("question", question);
    let response = await askOpenai(question, model);
    printFormatted("response", { candidate, response });
    return { response };
}

async function rebuttal(candidate, response, model) {
    let rebuttalPrompt = `O outro candidato respondeu: '${response}'. Por favor, forneça uma réplica.`;
    let rebuttal = await askOpenai(rebuttalPrompt, model);
    printFormatted("rebuttal", { candidate, rebuttal });
    return { rebuttal };
}

function saveResponses(responses) {
    fs.writeFileSync("responses.json", JSON.stringify(responses, null, 2));
}

async function evaluateResponses(responses) {
    let criteria = ["Clareza e Coerência", "Relevância", "Persuasão"];
    let scores = { "ChatGPT 4o Mini": [], "ChatGPT 3.5": [] };

    for (let response of responses) {
        for (let criterion of criteria) {
            let score4oMini = await evaluate(
                response["ChatGPT 4o Mini"],
                criterion
            );
            let score35 = await evaluate(response["ChatGPT 3.5"], criterion);
            scores["ChatGPT 4o Mini"].push(score4oMini);
            scores["ChatGPT 3.5"].push(score35);
        }
    }
    return scores;
}

async function evaluate(response, criterion) {
    const extractScore = (text) => {
        const regex = /(\d+(\.\d+)?)(\/10)?/;
        const match = text.match(regex);
        if (match) {
            return parseFloat(match[1]);
        }
        return 0;
    };

    let llamaScore = await askGroq(
        `Critério: ${criterion}. Avalie esta resposta: ${response.response}`,
        "llama3-8b-8192"
    );
    let geminiScore = await askGroq(
        `Critério: ${criterion}. Avalie esta resposta: ${response.response}`,
        "gemma-7b-it"
    );
    let mixtralScore = await askGroq(
        `Critério: ${criterion}. Avalie esta resposta: ${response.response}`,
        "mixtral-8x7b-32768"
    );

    let llamaScoreValue = extractScore(llamaScore);
    let geminiScoreValue = extractScore(geminiScore);
    let mixtralScoreValue = extractScore(mixtralScore);

    let averageScore =
        (llamaScoreValue + geminiScoreValue + mixtralScoreValue) / 3;
    return averageScore;
}

function compileVotes(scores) {
    let total4oMini = scores["ChatGPT 4o Mini"].reduce((a, b) => a + b, 0);
    let total35 = scores["ChatGPT 3.5"].reduce((a, b) => a + b, 0);

    console.log(`Total ChatGPT 4o Mini: ${total4oMini}`);
    console.log(`Total ChatGPT 3.5: ${total35}`);

    if (total4oMini > total35) {
        console.log("O vencedor é ChatGPT 4o Mini!");
    } else {
        console.log("O vencedor é ChatGPT 3.5!");
    }
}

(async function main() {
    let responses = await moderator(questions);
    let scores = await evaluateResponses(responses);
    compileVotes(scores);
})();
