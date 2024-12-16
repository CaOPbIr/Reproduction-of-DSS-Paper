const { MongoClient } = require('mongodb');
const axios = require('axios');
const fs = require('fs');

// MongoDB连接配置
const MONGO_URI = 'mongodb://localhost:27017';
const DATABASES = ['cellphone', 'laptop', 'camera', 'lipstick', 'runningshoes', 'ricecooker'];

// Flask API地址
const LDA_API_URL = 'http://localhost:5000/process_data';
const SENTIMENT_API_URL = 'http://localhost:5001/analyze_sentiment';

// 数据读取模块
async function fetchDataFromDB(client, dbName) {
    console.log(`\n读取数据库: ${dbName}`);
    const db = client.db(dbName);
    const collections = await db.listCollections().toArray();
    
    const dbData = {
        comments: [],
        replies: [],
        documents: [] // 存储带ID的完整文档
    };
    
    for (const collectionInfo of collections) {
        const collectionName = collectionInfo.name;
        console.log(`正在从集合读取数据: ${collectionName}`);
        
        const collection = db.collection(collectionName);
        const documents = await collection.find({}, {
            projection: { _id: 1, Comment: 1, Replies: 1 }
        }).toArray();

        // 存储完整文档
        documents.forEach(doc => {
            if (doc.Comment || (doc.Replies && doc.Replies.length > 0)) {
                dbData.documents.push({
                    collection: collectionName,
                    ...doc
                });
            }
        });

        // 提取评论和回复用于LDA分析
        const comments = documents
            .filter(doc => doc.Comment)
            .map(doc => doc.Comment);

        const replies = documents
            .filter(doc => Array.isArray(doc.Replies))
            .flatMap(doc => doc.Replies)
            .filter(reply => reply);

        dbData.comments = dbData.comments.concat(comments);
        dbData.replies = dbData.replies.concat(replies);
    }
    
    return dbData;
}

// LDA分析模块
async function performLDAAnalysis(dbName, data) {
    if (data.comments.length === 0 && data.replies.length === 0) {
        console.log(`数据库 ${dbName} 没有有效数据用于LDA分析`);
        return {
            commentCount: 0,
            replyCount: 0,
            results: []
        };
    }

    try {
        const response = await axios.post(LDA_API_URL, {
            comments: data.comments,
            replies: data.replies
        });

        const result = {
            commentCount: data.comments.length,
            replyCount: data.replies.length,
            results: response.data.results
        };

        console.log(`\n${dbName} LDA分析结果：`);
        console.log('评论总数：', result.commentCount);
        console.log('回复总数：', result.replyCount);
        console.log('\n属性词分析结果：');
        result.results.forEach((item, index) => {
            console.log(`${index + 1}. ${item.word} (权重: ${item.weight}) - 来源: ${item.source.join(', ')}`);
        });
        console.log('='.repeat(50));

        return result;
    } catch (error) {
        console.error(`LDA分析 ${dbName} 时发生错误:`, error.message);
        throw error;
    }
}

// 情感分析模块
async function performSentimentAnalysis(dbName, data) {
    const results = [];
    
    for (const doc of data.documents) {
        try {
            const response = await axios.post(SENTIMENT_API_URL, doc);
            if (response.data.status === 'success') {
                results.push({
                    collection: doc.collection,
                    ...response.data.result
                });
            }
        } catch (error) {
            console.error(`情感分析文档 ${doc._id} 时发生错误:`, error.message);
        }
    }
    
    return results;
}

// 处理单个数据库
async function processDatabase(client, dbName, options) {
    const data = await fetchDataFromDB(client, dbName);
    const results = {};
    
    if (options.lda) {
        results.lda = await performLDAAnalysis(dbName, data);
    }
    
    if (options.sentiment) {
        results.sentiment = await performSentimentAnalysis(dbName, data);
    }
    
    return results;
}

// 处理所有数据库
async function processAllDatabases(options = { lda: false, sentiment: false }) {
    let client;
    try {
        client = await MongoClient.connect(MONGO_URI);
        const allResults = {};
        
        for (const dbName of DATABASES) {
            allResults[dbName] = await processDatabase(client, dbName, options);
        }

        // 保存结果
        if (options.lda) {
            fs.writeFileSync(
                'lda_results.json', 
                JSON.stringify(
                    Object.fromEntries(
                        Object.entries(allResults).map(([k, v]) => [k, v.lda])
                    ), 
                    null, 
                    2
                ), 
                'utf8'
            );
            console.log('\nLDA分析结果已保存到 lda_results.json');
        }
        
        if (options.sentiment) {
            fs.writeFileSync(
                'sentiment_results.json', 
                JSON.stringify(
                    Object.fromEntries(
                        Object.entries(allResults).map(([k, v]) => [k, v.sentiment])
                    ), 
                    null, 
                    2
                ), 
                'utf8'
            );
            console.log('\n情感分析结果已保存到 sentiment_results.json');
        }

        return allResults;
    } catch (error) {
        console.error('处理过程中发生错误：', error);
        throw error;
    } finally {
        if (client) {
            await client.close();
        }
    }
}

// 主函数
async function main() {
    try {
        // 可以通过参数控制执行哪些分析
        const options = {
            lda: process.argv.includes('--lda'),
            sentiment: process.argv.includes('--sentiment')
        };

        // 如果没有指定任何选项，默认执行所有分析
        if (!options.lda && !options.sentiment) {
            options.lda = true;
            options.sentiment = true;
        }

        console.log('开始数据处理...');
        console.log('执行的分析：', 
            [
                options.lda && 'LDA分析',
                options.sentiment && '情感分析'
            ].filter(Boolean).join(', ')
        );
        
        await processAllDatabases(options);
        
    } catch (error) {
        console.error('处理过程中发生错误：', error);
    }
}

// 如果直接运行此文件
if (require.main === module) {
    main();
}

module.exports = {
    fetchDataFromDB,
    performLDAAnalysis,
    performSentimentAnalysis,
    processAllDatabases
};