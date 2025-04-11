const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config(); // Load environment variables
const cors = require('cors'); // Import CORS middleware


const app = express();
const port = process.env.PORT || 4000;
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY)
const model = genAI.getGenerativeModel({model: 'gemini-1.5-flash'});


// Middleware to parse JSON request bodies
app.use(express.json());
app.use(cors())



// Endpoint to query Google Gemini with a drug name
app.post('/api/query-gemini-info', async (req, res) => {
 const { drugName } = req.body; // Get the drug name from the request body


 if (!drugName) {
   return res.status(400).json({ error: 'Drug name is required' });
 }


 const prompt = `Write me a brief description of this drug and what it's used for. Max 50 words. The drug is ${drugName}`;
 try {
   const result = await model.generateContent([prompt])
   console.log(result.response.text());
   console.log(`Querying Gemini for drug: ${drugName}`); // Debugging log
   return res.json({ data: result.response.text()})


 } catch (error) {
   console.error('Error querying the Gemini API:', error); // Log the full error for better debugging
   return res.status(500).json({ error: 'Failed to query Gemini API', details: error.message });
 }
});

app.post('/api/query-gemini-interactions', async (req, res) => {
  const { drugList } = req.body; // Get the drug name from the request body
 
 
  if (!drugList) {
    return res.status(400).json({ error: 'Drug name is required' });
  }
 
 
  const prompt = `I am inquiring about this drug list ${drugList}. For each drug, if you are able to provide me with possible symptoms,
  please do so in drug: symptoms format (please only list top four symptoms). If you cannot find any symptoms or it is is unknown, DO NOT WRITE ANYTHING row format: drug, symptoms. Please wrap the drug names in html <strong> tags.  If it's not a commonly used drug, please just state that.
  I'm aware that this list is for general knowledge and should not be considered medical advice, you don't have to write it.  Please
  omit any other commentary. Additionally, I'd like to see the increased risk of side effects when these drugs are taken together. This could be a paragraph. If there is just one drug entered, don't include this extra paragraph.`;
  try {
    const result = await model.generateContent([prompt])
    console.log(result.response.text());
    console.log(`Querying Gemini for drug: ${drugList}` ); // Debugging log
    return res.json({ data: result.response.text()})
 
 
  } catch (error) {
    console.error('Error querying the Gemini API:', error); // Log the full error for better debugging
    return res.status(500).json({ error: 'Failed to query Gemini API', details: error.message });
  }
 });


// Start server
app.listen(port, () => {
 console.log(`Server running on http://localhost:${port}`);
});




