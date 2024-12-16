const { MongoClient } = require('mongodb');
const fs = require('fs').promises;

const url = 'mongodb://localhost:27017';

const dbNames = ['cellphone', 'laptop', 'camera', 'lipstick', 'runningshoes', 'ricecooker'];


async function findReplies(collection) {
  try {
    const findResult = await collection.find({
      $and: [
        { Replies: { $ne: "No replies" } },
        { Replies: { $ne: [] } }
      ]
    }).toArray();
    await logToFile(`${findResult.length} replies found`);
    return findResult.map(doc => {
      return {
        _id: doc._id,
        Replies: doc.Replies
      }
    });
  } catch (err) {
    await logToFile(`Error finding documents: ${err}`);
    throw err;
  }
}

async function WriteToFile(logContent, filePath, append = false) {
  try {
    if (append) {
      await fs.appendFile(filePath, logContent);
    } else {
      await fs.writeFile(filePath, logContent);
    }
  } catch (err) {
    console.error('Error writing log to file:', err);
  }
}

function RepliesPreprocess(docs) {
  const regex = /[\u4e00-\u9fa5a-zA-Z]/;
  return docs.map(doc => {
    const processedReplies = doc.Replies.filter(item => regex.test(item));
    const hasChanged = JSON.stringify(processedReplies) !== JSON.stringify(doc.Replies);
    
    if (processedReplies.length > 0 && hasChanged) {
      return {
        _id: doc._id,
        Replies: processedReplies
      }
    }
    return null;
  }).filter(item => item !== null);
}

async function UpdateDatabase(collection, data, currentCount) {
  let updatedCount = currentCount;
  for (const item of data) {
    const filter = { _id: item._id };
    const update = { $set: { Replies: item.Replies } };
    await collection.updateOne(filter, update);
    updatedCount++;
    await logToFile(`Updated document ${item._id}, current count: ${updatedCount}`);
  }
  return updatedCount;
}

async function updateReplies(collection) {

  const filter = { Replies: "No replies" };
  const update = { $set: { Replies: [] } };

  const result = await collection.updateMany(filter, update);
  console.log(`${result.matchedCount} documents matched the filter, ${result.modifiedCount} documents were modified.`);
}

async function logToFile(message) {
  const timestamp = new Date().toISOString();
  const logMessage = `${timestamp}: ${message}\n`;
  await WriteToFile(logMessage, 'preprocess_log.txt', true);
}

async function main() {
  let client;
  let totalCount = 0;
  let collectionsProcessed = 0;
  try {
    client = await MongoClient.connect(url);
    await logToFile('Connected successfully to server');

    for (const dbName of dbNames) {
      const database = client.db(dbName);
      const collectionNames = await database.listCollections().toArray();
      await logToFile(`Processing database: ${dbName}`);
      
      for (const collectionName of collectionNames) {
        const collection = database.collection(collectionName.name);
        await logToFile(`Processing collection: ${collectionName.name}`);
        
        const foundReplies = await findReplies(collection);
        const processedReplies = RepliesPreprocess(foundReplies);
        totalCount = await UpdateDatabase(collection, processedReplies, totalCount);
        collectionsProcessed++;
      }
    }
    await logToFile(`Total ${collectionsProcessed} collections processed`);
    await logToFile(`Total ${totalCount} documents updated`);
  } catch (err) {
    await logToFile(`Error: ${err}`);
  } finally {
    if (client) {
      await client.close();
      await logToFile('Database connection closed');
    }
  }
}

async function test() {
  let client;
  let count = 0;
  try {
    client = await MongoClient.connect(url);
    console.log('Connected successfully to server');

    const database = client.db('cellphone');
    const collection = database.collection('4270019');
    const foundReplies = await findReplies(collection);
    const processedReplies = RepliesPreprocess(foundReplies);
    await UpdateDatabase(collection, processedReplies, count);
  } catch (err) {
    console.error(err);
  } finally {
    if (client) {
      await client.close();
    } 
  }
}

// test().catch(console.error);
main().catch(console.error);