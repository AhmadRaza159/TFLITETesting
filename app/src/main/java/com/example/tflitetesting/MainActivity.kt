package com.example.tflitetesting

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import com.example.tflitetesting.databinding.ActivityMainBinding
import com.example.tflitetesting.ml.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {
    private var bitmap:Bitmap?=null
    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding=ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnGallery.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_PICK_IMAGES)
            intent.type = "image/*" // or "image/*"
            startActivityForResult(intent, 1)
        }

        binding.btnProcess1.setOnClickListener {
            process(bitmap)
        }
        binding.btnProcess2.setOnClickListener {
            process2(bitmap)
        }
        binding.btnProcess3.setOnClickListener {
            process3(bitmap)
        }
        binding.btnProcess4.setOnClickListener {
            process4(bitmap)
        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == 1) {
                binding.img.setImageURI(data?.data)
                bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, data?.data)
            }
        }
    }

    private fun process(bitmap:Bitmap?){
        val model = Selfie2anime.newInstance(this)

// Creates inputs for reference.
        val selfieImage = TensorImage.fromBitmap(bitmap)

// Runs model inference and gets result.
        val outputs = model.process(selfieImage)
        val animeImage = outputs.animeImageAsTensorImage
        val animeImageBitmap = animeImage.bitmap

        binding.img.setImageBitmap(animeImageBitmap)

// Releases model resources if no longer used.
        model.close()
    }


    private fun process2(bitmap:Bitmap?){
        if (bitmap!=null){
            val model = LiteModelCartoonganFp161.newInstance(this)

// Creates inputs for reference.
            val sourceImage = TensorImage.fromBitmap(bitmap)

// Runs model inference and gets result.
            val outputs = model.process(sourceImage)
            val cartoonizedImage = outputs.cartoonizedImageAsTensorImage
            val cartoonizedImageBitmap = cartoonizedImage.bitmap

            binding.img.setImageBitmap(cartoonizedImageBitmap)

// Releases model resources if no longer used.
            model.close()
        }
    }
    private fun process3(bitmap:Bitmap?){
        if (bitmap!=null){
            val model = Stagelight.newInstance(this)

// Creates inputs for reference.
            val image = TensorImage.fromBitmap(bitmap)

// Runs model inference and gets result.
            val outputs = model.process(image)
            val imageOut = outputs.imageOutAsTensorImage
            val imageOutBitmap = imageOut.bitmap
            binding.img.setImageBitmap(imageOutBitmap)

// Releases model resources if no longer used.
            model.close()

        }
    }

    private fun process4(bitmap: Bitmap?){
        if (bitmap!=null){
//            val floatArrat= ByteArray(1f,2f,2f,4f,5f,6f,7f,8f,1f,2f,2f,4f,5f,6f,7f,8f,1f,2f,2f,4f,5f,6f,7f,8f)
            val floatArrat= ByteArray(24)
//            floatArrat.p
            val byteBuffer = ByteBuffer.allocate(24)
            byteBuffer.put(0,1)
            byteBuffer.put(1,1)
            byteBuffer.put(2,1)
            byteBuffer.put(3,1)
            byteBuffer.put(4,1)
            byteBuffer.put(5,1)
            byteBuffer.put(6,1)
            byteBuffer.put(7,1)
            byteBuffer.put(8,1)
            byteBuffer.put(9,1)
            byteBuffer.put(10,1)
            byteBuffer.put(11,1)
            byteBuffer.put(12,1)
            byteBuffer.put(13,1)
            byteBuffer.put(14,1)
            byteBuffer.put(15,1)
            byteBuffer.put(16,1)
            byteBuffer.put(17,1)
            byteBuffer.put(18,1)
            byteBuffer.put(19,1)
            byteBuffer.put(20,1)
            byteBuffer.put(21,1)
            byteBuffer.put(22,1)
            byteBuffer.put(23,1)
            val model = TestModel1.newInstance(applicationContext)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 24), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

// Releases model resources if no longer used.
            model.close()

        }
    }
}