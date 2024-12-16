const { MongoClient } = require('mongodb');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

// MongoDB连接配置
const MONGO_URI = 'mongodb://localhost:27017';
const DATABASES = ['cellphone', 'laptop', 'camera', 'lipstick', 'runningshoes', 'ricecooker'];

// Flask API地址
const IH_API_URL = 'http://localhost:5002/calc_IH';

// 检查必要的文件是否存在
function checkRequiredFiles() {
    const requiredFiles = [
        'D:/Codes/DeepLearning/LDA_Sentiment/lda_results.json',
        'D:/Codes/DeepLearning/LDA_Sentiment/sentiment_results.json'
    ];
    
    for (const file of requiredFiles) {
        if (!fs.existsSync(file)) {
            throw new Error(`缺少必要的文件: ${file}`);
        }
    }
}

async function processDatabase(client, dbName) {
    console.log(`\n处理数据库: ${dbName}`);
    const db = client.db(dbName);
    const collections = await db.listCollections().toArray();
    
    let allDocuments = [];
    
    // 从所有集合收集数据
    for (const collectionInfo of collections) {
        const collectionName = collectionInfo.name;
        console.log(`正在从集合收集数据: ${collectionName}`);
        
        const collection = db.collection(collectionName);
        
        // 获取所需字段的数据
        const documents = await collection.find({}, {
            projection: {
                _id: 1,
                Comment: 1,
                Rating: 1,
                Is_plus: 1,
                Has_image: 1,
                Has_video: 1,
                Image_num: 1,
            }
        }).toArray();
        
        allDocuments = allDocuments.concat(documents);
    }
    
    try {
        // 调用Python API计算IH值
        const response = await axios.post(IH_API_URL, {
            dbname: dbName,
            documents: allDocuments
        });
        
        if (response.data.status === 'error') {
            throw new Error(response.data.message);
        }
        
        return response.data.results;
        
    } catch (error) {
        console.error(`处理数据库 ${dbName} 时发生错误:`, error.message);
        throw error;
    }
}

async function processAllDatabases() {
    let client;
    try {
        checkRequiredFiles();
        
        client = await MongoClient.connect(MONGO_URI);
        const allResults = {};
        
        for (const dbName of DATABASES) {
            console.log(`开始处理数据库: ${dbName}`);
            try {
                const results = await processDatabase(client, dbName);
                // 对每个文档的键进行重新排序
                allResults[dbName] = results.map(doc => {
                    const orderedDoc = {};
                    // 指定键的顺序
                    const keyOrder = [
                        '_id',
                        'IH_1', 'IH_2', 'IH_3', 'IH_4', 'IH_5',
                        'IH_6', 'IH_7', 'IH_8', 'IH_9', 'IH_10',
                        'IH_11', 'IH_12', 'IH_13', 'IH_14'
                    ];
                    
                    // 按照指定顺序重建对象
                    keyOrder.forEach(key => {
                        orderedDoc[key] = doc[key];
                    });
                    
                    return orderedDoc;
                });
                console.log(`完成处理数据库: ${dbName}`);
            } catch (error) {
                console.error(`处理数据库 ${dbName} 失败: ${error.message}`);
                allResults[dbName] = { error: error.message };
            }
        }
        
        // 创建结果目录（如果不存在）
        const resultsDir = 'results';
        if (!fs.existsSync(resultsDir)) {
            fs.mkdirSync(resultsDir, { recursive: true });
        }
        
        // 使用自定义replacer函数来保持键的顺序
        const replacer = (key, value) => value;
        
        // 保存结果到文件
        fs.writeFileSync(
            path.join(resultsDir, 'ih_results.json'),
            JSON.stringify(allResults, replacer, 2),
            'utf8'
        );
        
        console.log('\n所有IH计算结果已保存到 calc_IH_EH/results/ih_results.json');
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

// 执行函数
processAllDatabases();

module.exports = { processAllDatabases };
