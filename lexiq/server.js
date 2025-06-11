// server.js
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');
const path = require('path');
const rateLimit = require('express-rate-limit');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting configuration
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 120, // limit each IP to 10 requests per windowMs
  message: 'Too many requests from this IP, please try again after 15 minutes',
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});

// Apply rate limiting to all API routes
app.use('/api/', limiter);

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Initialize OpenAI with error handling
let openai;
try {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set in environment variables');
  }
  openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
} catch (error) {
  console.error('Error initializing OpenAI:', error.message);
  process.exit(1);
}

// Enhanced logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.originalUrl} ${res.statusCode} - ${duration}ms`);
  });
  next();
});

// Routes with improved error handling
app.post('/api/analyze', async (req, res) => {
  try {
    const { terms } = req.body;
    
    if (!terms) {
      return res.status(400).json({ 
        status: 'error',
        message: 'No terms provided',
        code: 'MISSING_TERMS'
      });
    }
    
    if (terms.length > 50000) {
      return res.status(400).json({ 
        status: 'error',
        message: 'Terms text is too long. Maximum 50,000 characters allowed.',
        code: 'TERMS_TOO_LONG'
      });
    }
    
    const analysis = await analyzeTerms(terms);
    res.json({
      status: 'success',
      ...analysis
    });
  } catch (error) {
    console.error('Error analyzing terms:', error);
    
    // Determine appropriate error response
    if (error.name === 'OpenAIError' || error.response?.status === 429) {
      return res.status(429).json({ 
        status: 'error',
        message: 'OpenAI API rate limit exceeded. Please try again later.',
        code: 'RATE_LIMIT_EXCEEDED'
      });
    } else if (error.response?.status === 400) {
      return res.status(400).json({ 
        status: 'error',
        message: 'Bad request to OpenAI API. Terms may contain prohibited content.',
        code: 'BAD_REQUEST'
      });
    } else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.status(503).json({ 
        status: 'error',
        message: 'Unable to connect to OpenAI service. Please try again later.',
        code: 'SERVICE_UNAVAILABLE'
      });
    }
    
    res.status(500).json({ 
      status: 'error',
      message: 'Error analyzing terms and conditions',
      code: 'INTERNAL_SERVER_ERROR'
    });
  }
});

app.post('/api/fetch-terms', async (req, res) => {
  try {
    const { url } = req.body;
    
    if (!url) {
      return res.status(400).json({ 
        status: 'error',
        message: 'No URL provided',
        code: 'MISSING_URL'
      });
    }
    
    // Validate URL format
    try {
      new URL(url);
    } catch (e) {
      return res.status(400).json({ 
        status: 'error',
        message: 'Invalid URL format',
        code: 'INVALID_URL'
      });
    }
    
    const terms = await fetchTermsFromUrl(url);
    
    if (!terms) {
      return res.status(404).json({ 
        status: 'error',
        message: 'Could not find terms and conditions on the provided URL',
        code: 'TERMS_NOT_FOUND'
      });
    }
    
    res.json({ 
      status: 'success',
      terms 
    });
  } catch (error) {
    console.error('Error fetching terms:', error);
    
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.status(503).json({ 
        status: 'error',
        message: 'Unable to connect to the specified URL',
        code: 'CANNOT_CONNECT'
      });
    } else if (error.response?.status === 403) {
      return res.status(403).json({ 
        status: 'error',
        message: 'Access to the specified URL is forbidden',
        code: 'ACCESS_FORBIDDEN'
      });
    } else if (error.response?.status === 404) {
      return res.status(404).json({ 
        status: 'error',
        message: 'The specified URL was not found',
        code: 'URL_NOT_FOUND'
      });
    } else if (error.code === 'ETIMEDOUT') {
      return res.status(504).json({ 
        status: 'error',
        message: 'Connection to the specified URL timed out',
        code: 'CONNECTION_TIMEOUT'
      });
    }
    
    res.status(500).json({ 
      status: 'error',
      message: 'Error fetching terms from URL',
      code: 'FETCH_ERROR'
    });
  }
});

// Serve the frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Analysis Function with enhanced error handling and timeouts
async function analyzeTerms(terms) {
  try {
    // Create a system prompt for the analysis
    const systemPrompt = `
  You are LexAI, an AI specialized in analyzing Terms and Conditions, Privacy Policies, and other legal documents.
  Your task is to carefully analyze the provided legal document and identify concerning clauses that might negatively impact users.

  For each concerning clause you find:
  1. Identify the specific excerpt from the document
  2. Assign a risk level (high, medium, low)
  3. Provide a clear explanation of why it's concerning
  4. Give it a concise, descriptive title

  Additionally, create an overall safety score from 0-100 using this systematic methodology:

SCORING FRAMEWORK
──────────────────
1. Initialize base score: 100 (assumes fair, user-protective terms)

2. Apply categorical deductions based on the expanded legal-risk taxonomy  
   (values = High / Medium / Low):
   – Data Privacy & Rights .................... −12 / −7 / −3  
   – Liability Limitations .................... −10 / −6 / −2  
   – Dispute Resolution Bias .................. −15 / −8 / −4  
   – Content / IP Overreach ................... −8  / −5 / −2  
   – Termination / Suspension ................. −8  / −5 / −2  
   – Transparency / Notice .................... −6  / −3 / −1  
   – Payment & Auto-Renewal ................... −8  / −5 / −2  
   – Security & Breach Obligations ............ −10 / −6 / −2  
   – Compliance / Regulatory Gaps ............. −12 / −7 / −3  
   – Service-Level / Performance .............. −6  / −4 / −1  
   – Indemnities & Risk-Shifting .............. −10 / −6 / −2  

   CATEGORY WEIGHTING RATIONALE  
   – Dispute Resolution Bias receives the highest penalty (−15) as it eliminates fundamental legal recourse.  
   – Compliance / Regulatory Gaps and Data Privacy are next (−12) due to statutory exposure.  
   – Transparency / Notice and Service-Level receive lighter penalties as disclosure or SLAs mitigate risk.

3. Granular severity multiplier for clustering **(apply after per-category totals are calculated):**  
   – 2 HIGH clauses in the same category → multiply that category’s total deduction by 1.1  
   – 3 or more HIGH clauses in the same category → multiply by 1.2

4. Overlapping categories (single clause triggers multiple buckets):  
   – Apply the highest applicable deduction for that clause **plus 50 %** of each additional applicable deduction.  
   – Perform this adjustment before step 3 multipliers.

5. Positive credits for user-protective clauses (maximum **+15 total**, may raise score back up to, but not above, 100):  
   +2 Explicit GDPR/CCPA compliance  
   +2 30-day prior notice **and** opt-out for term changes  
   +2 Pro-rated refunds on cancellation  
   +2 Industry-standard SLAs **with service-credits**  
   +2 Mutual (not one-way) indemnities  

6. Clamp final score to the 0–100 range, then apply sanity cap →  
   **If any HIGH-risk clause exists, final score cannot exceed 69.**

7. Validate score band:  
   – ≥85  Minimal risk, well-balanced rights (no HIGH clauses)  
   – 70–84 Manageable; review MEDIUM clauses (no HIGH clauses)  
   – 50–69 Elevated; at least one major red flag (may include HIGH)  
   – 30–49 High risk; multiple serious issues  
   – <30  Severe; user-hostile or non-compliant

RISK CLASSIFICATION CRITERIA  
HIGH   Significantly limits rights or creates statutory exposure  
MEDIUM Creates material disadvantage or ambiguity  
LOW    Minor, industry-standard limitation with mitigations

Return your analysis in **this JSON format only**:
{
  "score": number,
  "summary": "string",
  "concerningClauses": [
    {
      "title": "string",
      "risk": "high|medium|low",
      "excerpt": "string",
      "explanation": "string",
      "categories": ["Category Name"]
    }
  ],
  "positiveClauses": [
    {
      "title": "string",
      "credit": 2,
      "excerpt": "string",
      "explanation": "string"
    }
  ]
}

Focus on identifying issues related to:
– Data collection, usage, and sharing (Data Privacy & Rights)  
– Privacy concerns and regulatory compliance (Data Privacy & Rights, Compliance / Regulatory Gaps)  
– Rights limitations and one-sided terms (Liability Limitations, Indemnities & Risk-Shifting)  
– Hidden fees, surcharges, auto-renewal, or price-escalation clauses (Payment & Auto-Renewal)  
– Termination and suspension conditions (Termination / Suspension)  
– Content ownership and licensing (Content / IP Overreach)  
– Liability limitations and disclaimers (Liability Limitations)  
– Dispute resolution, arbitration, or class-action waivers (Dispute Resolution Bias)  
– Changes to terms and notifications (Transparency / Notice)  
– Security obligations and breach handling (Security & Breach Obligations)  
– Service levels, performance guarantees, or illusory SLAs (Service-Level / Performance)  
– Governing law and jurisdiction (Dispute Resolution Bias)`;
    
    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o-2024-05-13",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: terms }
        ],
        response_format: { type: "json_object" },
        max_tokens: 4096,
        temperature: 0.4,
      });
      
      
      
      // Parse the response
      const analysisText = completion.choices[0].message.content;
      let analysis;
      
      try {
        analysis = JSON.parse(analysisText);
      } catch (parseError) {
        console.error('Error parsing OpenAI response:', parseError);
        throw new Error('Invalid response format from OpenAI');
      }
      
      // Validate analysis structure
      if (!analysis.score || !analysis.summary || !Array.isArray(analysis.concerningClauses)) {
        throw new Error('Invalid analysis structure from OpenAI');
      }
      
      return analysis;
    } catch (error) {
            
      // Handle specific API errors
      if (error.name === 'AbortError') {
        throw new Error('OpenAI API request timed out');
      }
      
      // Handle rate limiting with exponential backoff retry
      if (error.status === 429) {
        console.log('Rate limited, retrying...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        return analyzeTerms(terms); // Retry once
      }
      
      throw error;
    }
  } catch (error) {
    console.error('Error in analysis:', error);
    throw error;
  }
}

// URL Fetching Function with better error handling
async function fetchTermsFromUrl(url) {
  try {
    // Add protocol if missing
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'https://' + url;
    }
    
    // Fetch the webpage with timeout
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      timeout: 15000 // 15 second timeout
    });
    
    const html = response.data;
    const $ = cheerio.load(html);
    
    // Try to find terms and conditions content with enhanced selectors
    const potentialSelectors = [
      // Common IDs
      '#terms-and-conditions', '#terms', '#termsOfService', '#tos', '#terms-of-service', '#legal', '#legal-terms',
      // Common classes
      '.terms-and-conditions', '.terms', '.termsOfService', '.tos', '.terms-of-service', '.legal', '.legal-terms',
      // Common headings and containers 
      'h1:contains("Terms")', 'h1:contains("Terms of Service")', 'h1:contains("Terms and Conditions")',
      'h2:contains("Terms")', 'h2:contains("Terms of Service")', 'h2:contains("Terms and Conditions")',
      'div:contains("Terms of Service")', 'section:contains("Terms of Service")'
    ];
    
    let termsText = '';
    let termsElement = null;
    
    for (const selector of potentialSelectors) {
      const element = $(selector);
      if (element.length > 0) {
        // Found a potential container, now get its content
        termsElement = element;
        break;
      }
    }
    
    if (termsElement) {
      // Get the text from the element
      termsText = termsElement.text().trim();
      
      // If the element itself has very little text, try to get content from parent or children
      if (termsText.length < 500) {
        // Try parent if it's a heading
        if (termsElement.is('h1, h2, h3, h4, h5, h6')) {
          termsText = termsElement.parent().text().trim();
        } 
        // Try children if it's a container
        else if (termsElement.children().length > 0) {
          termsText = '';
          termsElement.children('p, div, section, article, li').each((i, el) => {
            termsText += $(el).text() + '\n\n';
          });
          termsText = termsText.trim();
        }
      }
    }
    
    // If still no terms found, try body text as last resort
    if (termsText.length < 200) {
      const bodyText = $('body').text().trim();
      
      // Check if body text contains terms-related keywords
      const termsKeywords = ['terms of service', 'terms and conditions', 'user agreement', 'legal agreement'];
      const hasTermsKeywords = termsKeywords.some(keyword => bodyText.toLowerCase().includes(keyword));
      
      if (hasTermsKeywords) {
        termsText = bodyText;
      }
    }
    
    // Clean up the text
    termsText = termsText
      .replace(/\s+/g, ' ') // Replace multiple spaces/newlines with single space
      .replace(/\s+\./g, '.') // Fix spacing before periods
      .trim();
    
    // If text is still too short, it's probably not valid terms
    if (termsText.length < 500) {
      return null;
    }
    
    return termsText;
  } catch (error) {
    console.error('Error fetching URL:', error);
    throw error;
  }
}

// Start server with error handling
try {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`OpenAI API configured: ${!!process.env.OPENAI_API_KEY}`);
  });
} catch (error) {
  console.error('Error starting server:', error);
  process.exit(1);
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Application specific logging, throwing an error, or other logic here
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Application specific logging, throwing an error, or other logic here
  
  // For severe errors, exit the process after logging
  // This allows process managers like PM2 to restart the application
  process.exit(1);
});

module.exports = app; // Export for testing
