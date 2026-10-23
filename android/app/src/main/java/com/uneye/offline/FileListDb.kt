package com.uneye.offline

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * Room database for the app. Per the plan, Room stores ONLY the file list
 * (which recordings exist, where their labels live, their load settings) —
 * the labels themselves live in the C++ ExpertLabels (saved exactly through
 * Native.saveLabels).
 */
@Entity(tableName = "trace_files")
data class TraceFileEntity(
    @PrimaryKey val uri: String,          // content:// uri of the trace file (.mat, or the X csv)
    val yUri: String,                     // Y csv uri (empty for .mat)
    val displayName: String,
    val rateHz: Double,                   // declared sampling rate
    val degPerUnit: Double,               // units -> degrees
    val labelsPath: String,               // where ExpertLabels are saved (per trial: path + ".trialK")
    val modelPath: String,                // last trained model for this file (empty = none yet)
    val addedAt: Long = System.currentTimeMillis(),
)

@Dao
interface TraceFileDao {
    @Query("SELECT * FROM trace_files ORDER BY addedAt DESC")
    suspend fun all(): List<TraceFileEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(file: TraceFileEntity)

    @Delete
    suspend fun delete(file: TraceFileEntity)

    @Query("SELECT * FROM trace_files WHERE uri = :uri")
    suspend fun get(uri: String): TraceFileEntity?

    @Query("UPDATE trace_files SET labelsPath = :labelsPath, modelPath = :modelPath WHERE uri = :uri")
    suspend fun updatePaths(uri: String, labelsPath: String, modelPath: String)
}

@Database(entities = [TraceFileEntity::class], version = 1, exportSchema = false)
abstract class FileListDb : RoomDatabase() {
    abstract fun traceFiles(): TraceFileDao

    companion object {
        @Volatile private var instance: FileListDb? = null

        fun get(context: Context): FileListDb = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext, FileListDb::class.java, "trace_files.db"
            ).build().also { instance = it }
        }
    }
}
