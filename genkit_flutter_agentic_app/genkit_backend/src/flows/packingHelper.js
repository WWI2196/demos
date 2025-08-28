import { z } from 'genkit';
import { ai } from "../ai.js";
import { getWeatherTool } from "../tools/weather.js";
import { getLatLongTool } from "../tools/latlong.js";
import { getUnsplashTool } from '../tools/unsplash.js';
import { gemini20Flash } from '@genkit-ai/googleai';

//  INPUT Schema from client
const PackingHelperInputSchema = z.object({
    numberOfDays: z.number().describe('Number of days that the traveler is traveling'),
    location: z.string().describe('The location where the traveler is going'),
    preferences: z.string().describe('The traveler\'s packing preferences when it comes to what they like to wear / pack for their vacation'),
});

// Intermediary Schema
// Data scheme for weather forecast in a given location
const WeatherSchema = z.object({
    location: z.object({
        city: z.string().describe('The name of the city where the traveler is going'),
        state: z.string().describe('The name of the state or province where the traveler is going')
    }),
    weatherForecast: z.string().describe(`A 2 sentence summary of the weather forecast.`),
});

// Data scheme to store daily outfit plans.
const OutfitPlannerSchema = z.array(z.object({
    date: z.string().describe('The date this outfit should be worn.'),
    outfit: z.string('A description of the outfit to be worn with an itemized list of articles of clothing.')
}));

// OUTPUT Schema send to client
const ArticleOfClothingSchema = z.object({
    name: z.string().describe('Name of the article of clothing'),
    quantity: z.number().describe('Quantity of this item needed.'),
    dates: z.array(z.string().describe('dates when this article of clothing is applicable')),
    notes: z.string().optional().describe('Any additional notes for the user.'),
    optional: z.boolean().describe('Whether or not this item is optional')
});

const PackingChecklistSchema = z.object({
    items: z.array(ArticleOfClothingSchema).describe('The items that need to be packed'),
});

// PACKING HELPER FLOW
// Input: Provide the location that you're visiting, number of days, and any attire preferences.
// Output: Get the weather forecast, a hero image, and packing checklist for that location
export const packingHelperFlow = ai.defineFlow(
    {
        name: 'packingHelperFlow',
        inputSchema: PackingHelperInputSchema,
    },
    async (input) => {
        console.log('Packing helper flow started with input:', JSON.stringify(input, null, 2));

        let weatherResponse;
        let location;
        let weather;
        
        try {
            console.log('Starting weather generation...');
            weatherResponse = await ai.generate({
                prompt: `You are a travel weather assistant. Your task is to:
                1. Use the getLatLong tool to get coordinates for: ${input.location}
                2. Use the getWeather tool to get the ${input.numberOfDays}-day forecast for those coordinates
                3. Extract the city and state from the location: ${input.location}
                4. Create a 2-sentence weather summary
                
                Return a JSON object with this exact structure:
                {
                  "location": {
                    "city": "CityName",
                    "state": "StateName"
                  },
                  "weatherForecast": "Two sentence weather summary here."
                }`,
                tools: [getWeatherTool, getLatLongTool],
                model: gemini20Flash,
                output: {
                    schema: WeatherSchema,
                },
                config: {
                    temperature: 0.1, // Lower temperature for more consistent structured output
                }
            });

            console.log('Weather generation completed. Raw response:', JSON.stringify(weatherResponse, null, 2));

            if (!weatherResponse.output || !weatherResponse.output.location || !weatherResponse.output.weatherForecast) {
                throw new Error('Invalid weather response structure');
            }

            location = weatherResponse.output.location;
            weather = weatherResponse.output.weatherForecast;
            console.log('Extracted location:', location, 'weather:', weather);
        } catch (error) {
            console.error('Weather generation failed:', error);
            // Fallback: parse location manually and provide default weather
            const locationParts = input.location.split(',').map(part => part.trim());
            location = {
                city: locationParts[0] || 'Unknown City',
                state: locationParts[1] || 'Unknown State'
            };
            weather = `Weather forecast for ${location.city}, ${location.state} for the next ${input.numberOfDays} days. Please check local weather services for current conditions.`;
            console.log('Using fallback location:', location, 'weather:', weather);
        }

        const outfitsResponse = await ai.generate({
            system: 'You are an expert personal stylist. A traveler has asked you to put together outfits for them to wear for an upcoming travel trip. You curate outfits based on the weather and the traveler\'s preferences.',
            prompt: `Using your fashion expertise, please create an itemized list of ${input.numberOfDays} outfits that the user should pack for the weather forecast: ${weather} and matches the traveler's attire preferences: ${input.preferences}. 
            Follow these rules: 
            - Every outfit needs to have at least a top (tshirt, shirt, sweater.) and bottom (pants, jeans, sweatpants, skirt.) unless the outfit is a dress. 
            - Layers of clothes are allowed, so the traveler can wear jackets, cardigans, hoodies, etc. 
            - If it's cold or raining, the traveler will need a jacket that can be worn for multiple days.
            - If it's sunny, suggest sunglasses and a hat.`,
            config: {
                maxOutputTokens: 400,
                stopSequences: ['<end>', '<fin>'],
                temperature: 1.2,
                topP: 0.4,
                topK: 50,
            },
            output: { schema: OutfitPlannerSchema },
        });

        const outfits = outfitsResponse.output;

        const checklistResponse = await ai.generate({
            prompt: `Analyze this list of outfits: ${JSON.stringify(outfits)} and create a consolidated checklist of items that the user needs to pack. Aggregate common items and provide the exact item count needed.`,
            output: { schema: PackingChecklistSchema },
            model: gemini20Flash,
            config: {
                temperature: 1,
            },
        });

        const checklist = checklistResponse.output;

        const heroImageResponse = await ai.generate({
            prompt: `Get JUST the URL to an image of the travel destination: ${input.location} as a string. Use a search query that wil return an image of a popular tourist attraction in ${location.city},${location.state}. If there is no famous attraction, find an image of the ${location.city} skyline.`,
            tools: [getUnsplashTool],
            model: gemini20Flash,
            output: z.string().describe(`URL to an image of ${input.location}`)
        });

        const heroImage = heroImageResponse.text;

        return {
            location: location,
            lengthOfStay: input.numberOfDays,
            weather: weather,
            heroImage: heroImage,
            checklist: checklist
        };
    });
